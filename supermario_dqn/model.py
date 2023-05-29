import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelDQN(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(DuelDQN, self).__init__()
        
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")
        
        self.mid_network = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 32),
        )
        
        self.advantage = nn.Linear(32, output_dim)
        self.value = nn.Linear(32, 1)
    
    def forward(self, x):
        
        mid = self.mid_network(x)
        
        value = self.value(mid)
        advantage = self.advantage(mid)
        
        advAverage = torch.mean(advantage, dim=1, keepdim=True)
        q_value = value + advantage - advAverage
        
        return q_value