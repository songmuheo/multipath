# agents/rdqn_agent.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents.base_agent import BaseAgent
from models.rdqn_model import RecurrentDQNNetwork
from collections import deque

class RDQNAgent(BaseAgent):
    def __init__(self, state_size, action_size, config):
        super(RDQNAgent, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.state_size = state_size
        self.action_size = action_size

        # 하이퍼파라미터 로드
        agent_params = config
        self.gamma = agent_params['gamma']
        self.epsilon = agent_params['epsilon_start']
        self.epsilon_min = agent_params['epsilon_min']
        self.epsilon_decay = agent_params['epsilon_decay']
        self.learning_rate = agent_params['learning_rate']
        self.batch_size = agent_params['batch_size_dqn']
        self.memory_size = agent_params['memory_size']
        self.sequence_length = agent_params['sequence_length']
        self.burn_in_length = agent_params['burn_in_length']

        # 메모리 초기화 (deque 사용)
        self.memory = deque(maxlen=self.memory_size)

        # 모델 정의
        self.model = RecurrentDQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_model = RecurrentDQNNetwork(self.state_size, self.action_size).to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss(reduction='none')  # 손실 마스킹을 위해 reduction='none' 설정

        self.is_eval_mode = False

        self.target_model_update_freq = agent_params['target_model_update_freq']
        self.update_counter = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """
        메모리에 (state, action, reward, next_state, done) 형태의 전이를 저장합니다.
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample_sequences(self):
        """
        메모리에서 시퀀스를 샘플링합니다.
        """
        total_length = self.sequence_length + self.burn_in_length
        if len(self.memory) < total_length:
            return None

        # 에피소드 경계를 고려하여 인덱스 생성
        episodes = []
        episode = []
        for transition in self.memory:
            episode.append(transition)
            if transition[4]:  # done 플래그가 True이면 에피소드 종료
                episodes.append(episode)
                episode = []
        if episode:
            episodes.append(episode)

        # 충분한 길이의 에피소드 선택
        valid_episodes = [ep for ep in episodes if len(ep) >= total_length]
        if not valid_episodes:
            return None

        # 배치 구성
        sequences = []
        for _ in range(self.batch_size):
            ep = random.choice(valid_episodes)
            idx = random.randint(0, len(ep) - total_length)
            sequence = ep[idx:idx + total_length]
            sequences.append(sequence)

        return sequences

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

    def select_action(self, state, hidden_state):
        """
        현재 상태에서 행동을 선택합니다.
        """
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, state_size)
        if hidden_state is None:
            hidden_state = self.model.init_hidden(batch_size=1)

        with torch.no_grad():
            q_values, new_hidden_state = self.model(state, hidden_state)
            q_values = q_values.squeeze(0)  # (1, action_size)

        if self.is_eval_mode:
            action = torch.argmax(q_values).item()
        elif np.random.rand() > self.epsilon:
            action = torch.argmax(q_values).item()
        else:
            action = random.randrange(self.action_size)
        max_q_value = torch.max(q_values).item()
        return action, max_q_value, new_hidden_state

    def replay(self):
        """
        메모리에서 배치를 샘플링하여 모델을 학습합니다.
        """
        if self.is_eval_mode:
            return 0.0

        # 시퀀스 샘플링
        batch = self.sample_sequences()
        if batch is None:
            return 0.0

        # 배치 데이터 구성
        batch_size = len(batch)
        total_length = self.sequence_length + self.burn_in_length

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        masks = []

        for sequence in batch:
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            mask = []
            for i, (s, a, r, s_next, d) in enumerate(sequence):
                states.append(s)
                actions.append(a)
                rewards.append(r)
                next_states.append(s_next)
                dones.append(d)
                if i >= self.burn_in_length:
                    mask.append(1.0)
                else:
                    mask.append(0.0)
            state_batch.append(states)
            action_batch.append(actions)
            reward_batch.append(rewards)
            next_state_batch.append(next_states)
            done_batch.append(dones)
            masks.append(mask)

        # Tensor로 변환
        state_batch = torch.FloatTensor(state_batch).to(self.device)  # (batch_size, total_seq_len, state_size)
        action_batch = torch.LongTensor(action_batch).unsqueeze(-1).to(self.device)  # (batch_size, total_seq_len, 1)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)  # (batch_size, total_seq_len)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)  # (batch_size, total_seq_len, state_size)
        done_batch = torch.FloatTensor(done_batch).to(self.device)  # (batch_size, total_seq_len)
        masks = torch.FloatTensor(masks).to(self.device)  # (batch_size, total_seq_len)

        # 은닉 상태 초기화 (배치 내 독립적인 초기화)
        hidden_state = self.model.init_hidden(batch_size=batch_size)
        target_hidden_state = self.target_model.init_hidden(batch_size=batch_size)

        # Burn-in 기간 동안 은닉 상태 업데이트
        with torch.no_grad():
            _, hidden_state = self.model(state_batch[:, :self.burn_in_length, :], hidden_state)
            _, target_hidden_state = self.target_model(state_batch[:, :self.burn_in_length, :], target_hidden_state)

        # Burn-in 이후의 시퀀스에 대해 Q값 계산
        q_values, _ = self.model(state_batch[:, self.burn_in_length:, :], hidden_state)
        q_values = q_values.gather(2, action_batch[:, self.burn_in_length:, :]).squeeze(-1)  # (batch_size, seq_len)

        # 다음 상태의 최대 Q값 계산
        with torch.no_grad():
            next_q_values, _ = self.target_model(next_state_batch[:, self.burn_in_length:, :], target_hidden_state)
            next_max_q_values = next_q_values.max(2)[0]  # (batch_size, seq_len)

        # 목표 Q값 계산
        rewards = reward_batch[:, self.burn_in_length:]  # (batch_size, seq_len)
        dones = done_batch[:, self.burn_in_length:]  # (batch_size, seq_len)
        masks = masks[:, self.burn_in_length:]  # (batch_size, seq_len)

        targets = rewards + self.gamma * next_max_q_values * (1 - dones)

        # 손실 계산 (마스크 적용)
        loss_elements = self.criterion(q_values, targets)
        mask_sum = masks.sum()
        if mask_sum == 0:
            return 0.0  # 마스크의 합이 0이면 업데이트하지 않음
        loss = (loss_elements * masks).sum() / (mask_sum + 1e-8)

        # 모델 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping 적용
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        # 타겟 네트워크 업데이트
        self.update_counter += 1
        if self.update_counter % self.target_model_update_freq == 0:
            self.update_target_model()

        return loss.item()

    def set_train_mode(self):
        """학습 모드로 설정합니다."""
        self.model.train()
        self.is_eval_mode = False

    def set_eval_mode(self):
        """평가 모드로 설정합니다."""
        self.model.eval()
        self.is_eval_mode = True

    def save_model(self, file_path, episode):
        """모델의 가중치를 파일에 저장합니다."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'episode': episode
        }, file_path)

    def load_model(self, file_path):
        """저장된 모델 가중치를 불러옵니다."""
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])