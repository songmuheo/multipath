# agents/ppo_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.base_agent import BaseAgent
from collections import deque

class PPOAgent(BaseAgent):
    def __init__(self, state_size, action_size, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"device: {self.device}")
        self.config = config
        self.state_size = state_size
        self.action_size = action_size

        # 하이퍼파라미터 로드
        self.gamma = self.config['gamma']
        self.lambda_gae = self.config['lambda_gae']  # GAE 람다 추가
        self.epsilon_clip = self.config['epsilon_clip']
        self.k_epochs = self.config['k_epochs']
        self.learning_rate = self.config['learning_rate']
        self.entropy_coef = self.config['entropy_coef']
        self.value_loss_coef = self.config['value_loss_coef']
        self.batch_size = self.config['batch_size_ppo']
        self.mini_batch_size = self.config['mini_batch_size']
        self.update_timestep = config['update_timestep']  # 업데이트 주기 설정
        self.timestep = 0  # 현재 스텝 초기화

        # 모델 정의
        self.policy = self.build_model()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        # Scheduler 초기화
        self.scheduler_type = self.config['scheduler_type']
        self.scheduler = self._initialize_scheduler()
        self.policy_old = self.build_model()
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.policy.to(self.device)
        self.policy_old.to(self.device)

        self.memory = []

        self.is_eval_mode = False

    def _initialize_scheduler(self):
        """스케줄러 초기화"""
        if self.scheduler_type == "CosineAnnealingLR":
            T_max = self.config['scheduler_T_max']
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif self.scheduler_type == "LinearLR":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=self.config['start_factor'], end_factor=self.config['end_factor'], total_iters=self.config['total_iters'])
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")


    def build_model(self):
        network = self.config['network_ppo']
        if network == 'ActorCritic':
            from models.ppo_models import ActorCritic
            return ActorCritic(self.state_size, self.action_size)
        elif network == 'ActorCritic2':
            from models.ppo_models import ActorCritic2
            return ActorCritic2(self.state_size, self.action_size)
        elif network == 'ActorCritic_batchnorm':
            from models.ppo_models import ActorCritic_batchnorm
            return ActorCritic_batchnorm(self.state_size, self.action_size)
        elif network == 'ActorCritic2_batchnorm':
            from models.ppo_models import ActorCritic2_batchnorm
            return ActorCritic2_batchnorm(self.state_size, self.action_size)            
        else:
            raise ValueError(f"Unsupported model name: {network}")


    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.policy_old(state)  # 이전 정책 사용
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        if not self.is_eval_mode:
            self.memory.append({'state': state,
                                'action': action,
                                'action_log_prob': action_log_prob,
                                'reward': None,
                                'done': None})
        return action.item()

    def remember(self, reward, done):
        # 최근 메모리에 reward와 done 추가
        self.memory[-1]['reward'] = reward
        self.memory[-1]['done'] = done
        self.timestep += 1

        # 업데이트 주기에 도달하면 정책 업데이트
        loss = None
        if self.timestep % self.update_timestep == 0:
            loss = self.update()
            self.memory = []
        return loss

    def compute_gae(self, rewards, dones, values, next_value):
        gae = 0
        returns = []
        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        return returns
    def update(self):
        # 메모리에서 데이터 추출
        states = torch.cat([m['state'] for m in self.memory]).to(self.device)
        actions = torch.cat([m['action'] for m in self.memory]).to(self.device)
        rewards = [m['reward'] for m in self.memory]
        dones = [m['done'] for m in self.memory]
        old_action_log_probs = torch.cat([m['action_log_prob'] for m in self.memory]).to(self.device)

        # 현재 가치 예측
        with torch.no_grad():
            _, state_values = self.policy(states)
            state_values = state_values.squeeze().cpu().numpy()
        state_values = list(state_values)

        # 다음 상태 가치 예측
        if self.memory[-1]['done']:
            next_value = 0
        else:
            next_state = self.memory[-1]['state']
            with torch.no_grad():
                _, next_value = self.policy(next_state)
                next_value = next_value.item()

        # GAE를 사용하여 리턴 및 어드밴티지 계산
        returns = self.compute_gae(rewards, dones, state_values, next_value)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device).detach()
        state_values = torch.tensor(state_values, dtype=torch.float32).to(self.device)
        advantages = returns - state_values

        # 어드밴티지 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # 손실 추적용 리스트
        losses = []

        # Mini-Batch 학습
        total_data = len(states)
        for _ in range(self.k_epochs):
            indices = np.arange(total_data)
            np.random.shuffle(indices)
            for i in range(0, total_data, self.mini_batch_size):
                batch_indices = indices[i:i+self.mini_batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_action_log_probs = old_action_log_probs[batch_indices]

                # 배치에 대한 확률 및 가치 예측
                action_probs, state_values = self.policy(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                action_log_probs = dist.log_prob(batch_actions)
                dist_entropy = dist.entropy()

                # 확률 비 계산
                ratios = torch.exp(action_log_probs - batch_old_action_log_probs.detach())

                # 손실 함수 계산
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.value_loss_coef * nn.MSELoss()(state_values.squeeze(), batch_returns)
                entropy_loss = -self.entropy_coef * dist_entropy.mean()

                loss = actor_loss + critic_loss + entropy_loss

                # 손실 저장
                losses.append(loss.item())

                # 모델 업데이트
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # 오래된 정책 업데이트
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 평균 손실 반환
        return np.mean(losses)
    def update_scheduler(self):
        """스케줄러 업데이트 (에피소드 단위)"""
        self.scheduler.step()

    def get_current_lr(self):
        """현재 학습률 반환"""
        # 스케줄러에서 현재 학습률을 가져옴
        return self.scheduler.get_last_lr()[0]

    def set_train_mode(self):
        self.is_eval_mode = False
        self.policy.train()
        print("Agent set to training mode.")

    def set_eval_mode(self):
        self.is_eval_mode = True
        self.policy.eval()
        print("Agent set to evaluation mode.")

    def save_model(self, file_path, episode):
        torch.save({
            'episode': episode,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, file_path)
        print(f"Model saved at episode {episode + 1} -> {file_path}")

    def load_model(self, file_path):
        checkpoint = torch.load(file_path)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.start_episode = checkpoint['episode'] + 1  # 다음 에피소드부터 시작
        print(f"Model loaded from {file_path}, starting from episode {self.start_episode}")