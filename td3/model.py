from typing import Union, Optional, Dict, Any
import numpy as np

import torch
from torch import nn

from tianshou.data import to_torch, to_torch_as
from tianshou.utils.net import continuous

class Cnn2d(nn.Module):

    def __init__(self, action_shape=0, num_frames=1, device="cpu"):
        super().__init__()
        self.device = device
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=num_frames, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512 + np.prod(action_shape), 256),
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
        s = s.reshape(s.size(0), -1, 84, 84)

        features = self.conv1(s)
        features = self.conv2(features)
        features = self.conv3(features)
        features = self.fc1(features)
        features = features.flatten(1)
        if a is not None:
            features = torch.cat([features, a], dim=1)
        features = self.fc2(features)

        return features, state


class Cnn2dDeep(nn.Module):

    def __init__(self, action_shape=0, num_frames=1, device="cpu"):
        super().__init__()
        self.device = device
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=num_frames, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 1024),
            nn.ReLU()
        )
        action_sahpe = np.prod(action_shape)
        if action_shape != 0:
            self.action_embedding = nn.Sequential(
                nn.Linear(np.prod(action_shape), 128),
                nn.ReLU()
            )
        self.fc2 = nn.Sequential(
            nn.Linear(1152 if action_shape != 0 else 1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
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
        s = s.reshape(s.size(0), -1, 84, 84)

        features = self.conv1(s)
        features = self.conv2(features)
        features = self.conv3(features)
        features = self.fc1(features)
        features = features.flatten(1)
        if a is not None:
            action_features = self.action_embedding(a)
            features = torch.cat([features, action_features], dim=1)
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
        elif a is not None and self.network in ["cnn1d", "cnn2d", "cnn2d_deep"]:
            a = to_torch_as(a, s)
            a = a.flatten(1)
            logits, h = self.preprocess(s, a)
        else:
            logits, h = self.preprocess(s)

        logits = self.last(logits)
        return logits


class Cnn1d(nn.Module):
    def __init__(self, action_shape=0, num_frames=1, device="cpu"):
        super().__init__()
        self.action_shape = action_shape
        self.device = device
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=num_frames, out_channels=64, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=4),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8, stride=4),
            nn.ReLU()
        )
        self.avg_pool = nn.AvgPool1d(kernel_size = 3)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512 + np.prod(action_shape), 256)

    def forward(
            self,
            s: Union[np.ndarray, torch.Tensor],
            a: Optional[Union[np.ndarray, torch.Tensor]] = None,
            state=None,
            info: Dict[str, Any]={},
        ) -> torch.Tensor:
        s = to_torch(s, device=self.device, dtype=torch.float32)
        s = s.reshape(s.size(0), -1, 721)

        y = self.conv1(s)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.avg_pool(y)
        y = y.flatten(1)
        y = self.fc1(y)
        y = y.flatten(1)
        if a is not None:
            y = torch.cat([y, a], dim=1)
        y = self.fc2(y)

        return y, state
