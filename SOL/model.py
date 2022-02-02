from abc import abstractmethod
from collections import OrderedDict

import gym
import torch.nn as nn
import torch as th
import numpy as np
from gym import spaces
from talib import abstract

from CONFIG import *

from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.seq_nn import *


class BaseFeature(nn.Module):
    def __init__(self, observation_space: gym.Space, out_dim=16):
        super(BaseFeature, self).__init__()
        in_dim = gym.spaces.utils.flatdim(observation_space)
        self.dense = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Mish()
        )
        self.features_dim=out_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.dense(observations)




class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()
    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = th.sqrt(criterion(x, y))
        return loss




# class BaseMesh(nn.Module):
#     def __init__(self, obs_space,out_dim: int = 64):
#         super(BaseMesh,self).__init__()
#         seq_len = obs_space.shape[0]
#         seq_width = obs_space.shape[1]
#         self.cnn = CNN(seq_len, seq_width)
#         self.redu_seq = self.cnn.seq
#         self.cat_width = self.cnn.feature_concat_dim
#         self.rnn = SeqLstm(self.redu_seq,self.cat_width, layer_num=1, out_seq=1, out_dim=out_dim)
#         self.features_dim = out_dim
#
#     def forward(self, obs: th.Tensor) -> th.Tensor:
#         cc = self.cnn(obs)
#         return self.rnn(cc)


class BaseMesh(nn.Module):
    def __init__(self, obs_space,out_dim: int = 64):
        super(BaseMesh,self).__init__()
        seq_len = obs_space.shape[0]
        seq_width = obs_space.shape[1]
        self.cnn = SeqCNN(seq_len, seq_width, out_dim)
        self.features_dim = out_dim

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.cnn(obs)


class CombinedModel(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict):
        super (CombinedModel, self).__init__ ()
        self.obs = BaseMesh (observation_space.spaces[OBS])

        self.features_dim = self.obs.features_dim
        # self.features_dim = self.obs.features_dim + self.base.features_dim
    def forward(self, observations: TensorDict) -> th.Tensor:
        obs = self.obs(observations[OBS])
        return obs




class OutterModel(nn.Module):

    def __init__(self, spec):
        observation_space = spec.data_space

        super(OutterModel, self).__init__()

        self.module = CombinedModel(observation_space)
        dim1 = self.module.features_dim
        dim2= int(dim1/2)

        self.agg = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.BatchNorm1d(dim2),
            nn.Mish(),
        )
        self.price = nn.Linear (dim2, spec.price_len)
        self.direction =  nn.Sequential(
            nn.Linear(dim2, spec.price_len),
            nn.Tanh())

    def forward(self, observations: TensorDict) -> th.Tensor:
        feature = self.module(observations)
        agg = self.agg(feature)
        return th.cat([self.price(agg), self.direction(agg)], dim=1)
