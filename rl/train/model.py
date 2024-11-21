# train/model.py

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        # self.initialize_weights()

    def forward(self, x):
        return self.model(x)

    # def initialize_weights(self):
    #     for m in self.model:
    #         if isinstance(m, nn.Linear):
    #             init.kaiming_uniform_(m.weight, nonlinearity='relu')
    #             init.uniform_(m.bias, -0.1, 0.1)


class AdvancedDQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(AdvancedDQNNetwork, self).__init__()
        # 입력층에서 첫 번째 은닉층으로
        self.fc1 = nn.Linear(state_size, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Batch Normalization
        
        # Residual Block 1
        self.res1_fc1 = nn.Linear(256, 256)
        self.res1_fc2 = nn.Linear(256, 256)
        
        # Residual Block 2
        self.res2_fc1 = nn.Linear(256, 128)
        self.res2_fc2 = nn.Linear(128, 256)
        
        # 출력층으로
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        # 입력층 처리
        x = F.relu(self.bn1(self.fc1(x)))
        
        # Residual Block 1
        res1 = F.relu(self.res1_fc1(x))
        res1 = self.res1_fc2(res1)
        x = F.relu(x + res1)  # Skip connection
        
        # Residual Block 2
        res2 = F.relu(self.res2_fc1(x))
        res2 = self.res2_fc2(res2)
        x = F.relu(x + res2)  # Skip connection
        
        # 출력층 처리
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x