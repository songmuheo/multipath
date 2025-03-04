# agents/ppo_agent_vector.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.base_agent import BaseAgent
from collections import deque

class PPOAgent(BaseAgent):
    def __init__(self, state_size, action_size, config):
        """
        action_size: 튜플 (path_size, frame_size)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"device: {self.device}")
        self.config = config
        self.state_size = state_size
        self.path_size, self.frame_size = action_size  # 벡터 방식: 두 축의 크기

        self.gamma = self.config['gamma']
        self.lambda_gae = self.config['lambda_gae']
        self.epsilon_clip = self.config['epsilon_clip']
        self.k_epochs = self.config['k_epochs']
        self.learning_rate = self.config['learning_rate']
        self.entropy_coef = self.config['entropy_coef']
        self.value_loss_coef = self.config['value_loss_coef']
        self.batch_size = self.config['batch_size_ppo']
        self.mini_batch_size = self.config['mini_batch_size']
        self.update_timestep = config['update_timestep']
        self.timestep = 0

        self.policy = self.build_model()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.scheduler_type = self.config['scheduler_type']
        self.scheduler = self._initialize_scheduler()
        self.policy_old = self.build_model()
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.policy.to(self.device)
        self.policy_old.to(self.device)

        self.memory = []
        self.is_eval_mode = False

    def _initialize_scheduler(self):
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
            from models.ppo_models_vector import ActorCritic
            return ActorCritic(self.state_size, (self.path_size, self.frame_size))
        elif network == 'ActorCritic2':
            from models.ppo_models_vector import ActorCritic2
            return ActorCritic2(self.state_size, (self.path_size, self.frame_size))
        else:
            raise ValueError(f"Unsupported model name: {network}")

    def select_action(self, state, deterministic=False):
        """
        Action 선택 시 deterministic=True이면, 확률 분포의 argmax를 사용하여 결정적(Deterministic)으로 액션을 선택합니다.
        기본값은 확률적(Stochastic) 정책(deterministic=False)입니다.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            (path_probs, frame_probs), _ = self.policy_old(state)

        if deterministic:
            # 결정적 정책: 가장 높은 확률의 액션 선택
            path_action = torch.argmax(path_probs, dim=-1)
            frame_action = torch.argmax(frame_probs, dim=-1)
        else:
            # 확률적 정책: Categorical 분포에서 샘플링
            path_dist = torch.distributions.Categorical(path_probs)
            frame_dist = torch.distributions.Categorical(frame_probs)
            path_action = path_dist.sample()
            frame_action = frame_dist.sample()

        action = (path_action.item(), frame_action.item())
        
        if not self.is_eval_mode:
            self.memory.append({
                'state': state,
                'path_action': path_action,
                'frame_action': frame_action,
                'acion_log_prob': (path_dist.log_prob(path_action) + frame_dist.log_prob(frame_action)) if not deterministic else None,
                'reward': None,
                'done': None
            })
        return action


    def remember(self, reward, done):
        self.memory[-1]['reward'] = reward
        self.memory[-1]['done'] = done
        self.timestep += 1

        loss = None
        if self.timestep % self.config['update_timestep'] == 0:
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
        states = torch.cat([m['state'] for m in self.memory]).to(self.device)
        path_actions = torch.tensor([m['path_action'].item() for m in self.memory]).to(self.device)
        frame_actions = torch.tensor([m['frame_action'].item() for m in self.memory]).to(self.device)
        rewards = [m['reward'] for m in self.memory]
        dones = [m['done'] for m in self.memory]
        old_action_log_probs = torch.stack([m['action_log_prob'] for m in self.memory]).to(self.device)

        with torch.no_grad():
            _, state_values = self.policy(states)
            state_values = state_values.squeeze().cpu().numpy()
        state_values = list(state_values)

        if self.memory[-1]['done']:
            next_value = 0
        else:
            next_state = self.memory[-1]['state']
            with torch.no_grad():
                _, next_value = self.policy(next_state)
                next_value = next_value.item()

        returns = self.compute_gae(rewards, dones, state_values, next_value)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device).detach()
        state_values = torch.tensor(state_values, dtype=torch.float32).to(self.device)
        advantages = returns - state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        losses = []
        total_data = len(states)
        for _ in range(self.config['k_epochs']):
            indices = np.arange(total_data)
            np.random.shuffle(indices)
            for i in range(0, total_data, self.config['mini_batch_size']):
                batch_indices = indices[i:i+self.config['mini_batch_size']]
                batch_states = states[batch_indices]
                batch_path_actions = path_actions[batch_indices]
                batch_frame_actions = frame_actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_action_log_probs = old_action_log_probs[batch_indices]

                (path_probs, frame_probs), state_values = self.policy(batch_states)
                path_dist = torch.distributions.Categorical(path_probs)
                frame_dist = torch.distributions.Categorical(frame_probs)
                new_path_log_probs = path_dist.log_prob(batch_path_actions)
                new_frame_log_probs = frame_dist.log_prob(batch_frame_actions)
                new_action_log_probs = new_path_log_probs + new_frame_log_probs

                ratios = torch.exp(new_action_log_probs - batch_old_action_log_probs.detach())
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.config['epsilon_clip'], 1 + self.config['epsilon_clip']) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.config['value_loss_coef'] * nn.MSELoss()(state_values.squeeze(), batch_returns)
                entropy_loss = -self.config['entropy_coef'] * (path_dist.entropy().mean() + frame_dist.entropy().mean())

                loss = actor_loss + critic_loss + entropy_loss
                losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        return np.mean(losses)

    def update_scheduler(self):
        self.scheduler.step()

    def get_current_lr(self):
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
        self.start_episode = checkpoint['episode'] + 1
        print(f"Model loaded from {file_path}, starting from episode {self.start_episode}")
