import torch
from torch import nn
from tianshou.data import to_torch
import numpy as np

class CNN(nn.Module):

    def __init__(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_shape=[8, 8], stride=[4, 4]),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_shape=[4, 4], stride=[2, 2]),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_shape=[3, 3], stride=[1, 1]),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1152, 256)
        )

    def forward(self, obs, state=None, info={}):
        obs = to_torch(obs, device=self.device, dtype=torch.float32)
        obs = obs.reshape(obs.size(0), 1, 84, 84)

        features = self.conv1(obs)
        features = self.conv2(features)
        features = self.conv3(features)
        features = self.fc(features)

        return features