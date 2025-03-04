# models/ppo_models_vector.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        """
        action_size: 튜플 (path_size, frame_size)
        """
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        # 두 개의 액션 헤드: 경로(path)와 프레임(frame)
        self.path_head = nn.Linear(hidden_size, action_size[0])
        self.frame_head = nn.Linear(hidden_size, action_size[1])
        self.value_head = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        path_logits = self.path_head(x)
        frame_logits = self.frame_head(x)
        # 각각의 로짓에 대해 소프트맥스 적용하여 확률 분포 생성
        path_probs = F.softmax(path_logits, dim=-1)
        frame_probs = F.softmax(frame_logits, dim=-1)
        value = self.value_head(x)
        return (path_probs, frame_probs), value

class ActorCritic2(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        """
        action_size: 튜플 (path_size, frame_size)
        ActorCritic2는 Residual Connection을 추가한 버전입니다.
        """
        super(ActorCritic2, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.residual_fc = nn.Linear(hidden_size, hidden_size)  # Residual 추가
        # 벡터 방식에 맞게 두 개의 액션 헤드 생성
        self.path_head = nn.Linear(hidden_size, action_size[0])
        self.frame_head = nn.Linear(hidden_size, action_size[1])
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        res = F.relu(self.residual_fc(x))
        x = x + res  # Residual Connection
        path_logits = self.path_head(x)
        frame_logits = self.frame_head(x)
        path_probs = F.softmax(path_logits, dim=-1)
        frame_probs = F.softmax(frame_logits, dim=-1)
        state_values = self.value_head(x)
        return (path_probs, frame_probs), state_values
