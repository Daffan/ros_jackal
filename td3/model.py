from typing import Union, Optional, Dict, Any

import torch
from torch import nn
from tianshou.data import to_torch, to_torch_as
import numpy as np
from tianshou.utils.net import continuous

class CNN(nn.Module):

    def __init__(self, action_shape=0):
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
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1152, 512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512 + action_shape, 256),
            nn.ReLU()
        )

    def forward(
            self,
            s: Union[np.ndarray, torch.Tensor],
            a: Optional[Union[np.ndarray, torch.Tensor]] = None,
            state=None,
            info: Dict[str, Any]={},
        ) -> torch.Tensor:
        s = to_torch(s, device=self.device, dtype=torch.float32)
        s = s.reshape(s.size(0), 1, 84, 84)

        features = self.conv1(obs)
        features = self.conv2(features)
        features = self.conv3(features)
        features = self.fc1(features)
        features = features.flatten(1)
        features = torch.cat([features, a])
        features = self.fc2(features)

        return features, state

class Critic(continuous.Critic):
    def __init__(self, *args, network="mlp", **kwargs):
        super().__init__(*args, **kwargs)
        self.network = network
    
    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        a: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:

        s = to_torch(s, device=self.device, dtype=torch.float32)
        if a is not None and self.network == "mlp":
            s = s.flatten(1)
            a = to_torch_as(a, s)
            a = a.flatten(1)
            s = torch.cat([s, a], dim=1)
            logits, h = self.preprocess(s)
        elif a is not None and self.network == "cnn":
            a = to_torch_as(a, s)
            a = a.flatten(1)
            logits, h = self.preprocess(s, a)
        else:
            logits, h = self.preprocess(s)

        logits = self.last(logits)
        return logits