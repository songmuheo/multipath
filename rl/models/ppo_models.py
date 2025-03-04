# models/ppo_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(ActorCritic, self).__init__()
        # 기존 2개의 은닉층에서 3개로 늘림
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)  # 추가된 은닉층
        # Actor 헤드
        self.action_head = nn.Linear(hidden_size, action_size)
        # Critic 헤드
        self.value_head = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # 활성화 함수 ReLU 사용
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  # 추가된 은닉층
        # Actor와 Critic의 출력
        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_probs, state_values

class ActorCritic2(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(ActorCritic2, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.residual_fc = nn.Linear(hidden_size, hidden_size)  # Residual 추가
        self.action_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        res = F.relu(self.residual_fc(x))
        x = x + res  # Residual Connection
        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_probs, state_values

class ActorCritic3(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(ActorCritic3, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.residual_fc = nn.Linear(hidden_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        # Weight Initialization
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.residual_fc.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.xavier_uniform_(self.value_head.weight)

    def forward(self, x):
        if self.training and x.size(0) > 1:  # BatchNorm 사용 조건
            x1 = self.dropout(F.relu(self.bn1(self.fc1(x))))
            x2 = self.dropout(F.relu(self.bn2(self.fc2(x1))))
        else:  # BatchNorm 생략
            x1 = self.dropout(F.relu(self.fc1(x)))
            x2 = self.dropout(F.relu(self.fc2(x1)))
        
        # Residual Connection
        res = F.relu(self.residual_fc(x2))
        x = x2 + res
        
        # Output Layers
        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_probs, state_values

import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic4(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(ActorCritic4, self).__init__()
        
        # Layer 1
        self.fc1 = nn.Linear(state_size, hidden_size)
        # Layer 2
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Layer 3
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        # Residual Layer
        self.residual_fc = nn.Linear(hidden_size, hidden_size)
        
        # Action and Value Heads
        self.action_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # First Layer
        x = F.relu(self.fc1(x))
        # Second Layer
        x = F.relu(self.fc2(x))
        # Third Layer
        x = F.relu(self.fc3(x))
        
        # Residual Connection
        res = F.relu(self.residual_fc(x))
        x = x + res  # Add residual connection
        
        # Output heads
        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_probs, state_values



class ActorCritic_batchnorm(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(ActorCritic_batchnorm, self).__init__()
        
        # 은닉층과 BatchNorm 추가
        self.fc1 = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # BatchNorm 추가
            nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # BatchNorm 추가
            nn.ReLU()
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # BatchNorm 추가
            nn.ReLU()
        )
        
        # Actor 헤드 (행동 확률 예측)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic 헤드 (상태의 가치 예측)
        self.value_head = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = self.fc1(x)  # 첫 번째 은닉층
        x = self.fc2(x)  # 두 번째 은닉층
        x = self.fc3(x)  # 세 번째 은닉층
        action_probs = self.action_head(x)  # Actor의 행동 확률
        state_values = self.value_head(x)  # Critic의 상태 가치
        return action_probs, state_values
    
class ActorCritic2_batchnorm(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(ActorCritic2_batchnorm, self).__init__()
        
        # 은닉층과 BatchNorm 추가
        self.fc1 = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # BatchNorm 추가
            nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # BatchNorm 추가
            nn.ReLU()
        )
        
        # Residual Connection 추가
        self.residual_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # BatchNorm 추가
            nn.ReLU()
        )
        
        # Actor 헤드 (행동 확률 예측)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic 헤드 (상태의 가치 예측)
        self.value_head = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = self.fc1(x)  # 첫 번째 은닉층
        x = self.fc2(x)  # 두 번째 은닉층
        
        # Residual 연결 (x + residual)
        res = self.residual_fc(x)  # Residual 연결 추가
        x = x + res  # Residual 합산
        
        action_probs = self.action_head(x)  # Actor의 행동 확률
        state_values = self.value_head(x)  # Critic의 상태 가치
        return action_probs, state_values