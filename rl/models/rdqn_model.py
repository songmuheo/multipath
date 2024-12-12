
# models/rdqn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentDQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128, num_layers=1):
        super(RecurrentDQNNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x, hidden):
        x = F.relu(self.fc1(x))
        output, hidden = self.lstm(x, hidden)
        q_values = self.fc2(output)
        return q_values, hidden

    def init_hidden(self, batch_size):
        """
        LSTM의 초기 은닉 상태를 생성합니다.
        """
        device = next(self.parameters()).device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)