# agents/agent.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from train.model import DQNNetwork
from train.model import AdvancedDQNNetwork
from utils.config import load_config
from collections import deque
from agents.base_agent import BaseAgent


class DQNAgent(BaseAgent):
    def __init__(self, state_size, action_size, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"device: {self.device}")
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.network = config['network']

        # 하이퍼파라미터 로드
        self.gamma = self.config['gamma']
        self.epsilon = self.config['epsilon_start']
        self.epsilon_min = self.config['epsilon_min']
        self.epsilon_decay = self.config['epsilon_decay']
        self.learning_rate = self.config['learning_rate']
        self.batch_size = self.config['batch_size']
        self.memory_size = self.config['memory_size']
        self.memory = deque(maxlen=self.memory_size)
        
        # 모델 정의
        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.criterion = nn.MSELoss()
        self.criterion = nn.SmoothL1Loss()

        self.target_model = self.build_model()
        self.update_target_model()

        self.model.to(self.device)
        self.target_model.to(self.device)

        self.is_eval_mode = False

    def build_model(self):
        if self.network == 'DQNNetwork':
            return DQNNetwork(self.state_size, self.action_size)
        if self.network == 'AdvancedDQNNetwork':
            return AdvancedDQNNetwork(self.state_size, self.action_size)
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
            
    def select_action(self, state):
        # eval mode 이거나, 탐색을 하지 않는 경우
        if self.is_eval_mode or np.random.rand() > self.epsilon:
            # 최대 Q 값을 가진 행동 선택
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.model.eval()
            with torch.no_grad():
                act_values = self.model(state_tensor)
            if not self.is_eval_mode:
                self.model.train()
            action = torch.argmax(act_values).item()
            max_q_value = torch.max(act_values).item()
            return action, max_q_value
        # 무작위 탐색
        else:
            # 무작위 행동 선택 (탐색)
            return random.randrange(self.action_size), None


    def replay(self):
        # eval mode에서는 학습하지 않음
        if self.is_eval_mode:
            return
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([m[0] for m in minibatch]).to(self.device)
        actions = torch.LongTensor([m[1] for m in minibatch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([m[2] for m in minibatch]).to(self.device)
        next_states = torch.FloatTensor([m[3] for m in minibatch]).to(self.device)
        dones = torch.FloatTensor([m[4] for m in minibatch]).to(self.device)

        # 현재 상태의 Q값
        q_values = self.model(states).gather(1, actions).squeeze()

        # 다음 상태의 최대 Q값 (타겟 네트워크 사용)
        next_q_values = self.target_model(next_states).max(1)[0].detach()

        # 목표 Q값 계산
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        # 손실 계산
        loss = self.criterion(q_values, targets)

        # 모델 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def set_train_mode(self):
        """학습 모드로 설정합니다."""
        self.model.train()
        self.is_eval_mode = False
        print("Agent set to training mode.")

    
    def set_eval_mode(self):
        """평가 모드로 설정합니다."""
        self.model.eval()
        self.is_eval_mode = True
        print("Agent set to evaluation mode.")

    def save_model(self, file_path, episode):
        """모델의 가중치를 파일에 저장합니다."""
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved at episode {episode + 1} -> {file_path}")

    def load_model(self, file_path):
        """저장된 모델 가중치를 불러옵니다."""
        self.model.load_state_dict(torch.load(file_path))
        print(f"Model loaded from {file_path}")