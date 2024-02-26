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

class PPO(nn.Module):
    def __init__(self, state_shape, n_actions):
        super(PPO,self).__init__()
        '''
        입력변수
            state_shape: state 차원 -> [위치, 속도, 각도, 각속도]
            output_dim: actor 차원 -> [왼쪽, 오른쪽]
                        critic 차원 -> 1
            device : cpu, cuda device정보 
        N.N 구조
            2 - hidden layers, 64 nodes
            Activation function -> Relu
        '''
        self.state_shape = state_shape
        self.n_actions = n_actions
        
        self.seq = nn.Sequential(
            nn.Linear(self.state_shape,64), 
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
        )
        self.actor = nn.Linear(64,n_actions)
        self.critic = nn.Linear(64,1)

    def forward(self, state_t):
        policy = self.actor(self.seq(state_t))
        value = self.critic(self.seq(state_t))
        return policy, value

    def sample_actions(self,state_t):
        policy,_ = self.forward(state_t)
        policy = torch.squeeze(policy)
        softmax_policy = F.softmax(policy,dim=-1)
        softmax_policy = torch.nan_to_num(softmax_policy)
        action = torch.distributions.Categorical(softmax_policy).sample().item()
        return action
