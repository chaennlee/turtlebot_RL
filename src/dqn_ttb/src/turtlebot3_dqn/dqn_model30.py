#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(nn.Module):
    # in = 30, out = 5
    def __init__(self, state_size, action_size):
        super().__init__()

        self.in_layer = nn.Linear(state_size, 64)
        self.h_layer = nn.Linear(64, 64)
        self.out_layer = nn.Linear(64, action_size)

        self.act = nn.ReLU()

    def forward(self, x):
        ## 마지막 레이어는 ReLu를 사용하지 않는다. 사용한다면 출력값이 0~1 사이로 제한된다.
        h1 = self.act(self.in_layer(x))
        h2 = self.act(self.h_layer(h1))
        output = self.out_layer(h2)

        return output
