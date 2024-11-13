# train/model.py

import torch.nn as nn
import torch.nn.init as init

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
