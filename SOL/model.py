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
from stable_baselines3.seq_nn import SeqFeature, SeqLstm, SeqCRNN, SeqCNN, CNN


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

class ObsNN(nn.Module):
    def __init__(self, space):
        super(ObsNN,self).__init__()
        outdim: int = 32
        # self.parallels = [SeqLstm(space, out_dim=outdim), SeqCRNN(space, out_dim=outdim), SeqCNN(space, out_dim=outdim)]
        self.parallels = [SeqCRNN(space, out_dim=outdim)]
        self.parallels = nn.ModuleList(self.parallels)
        self.features_dim = outdim * len(self.parallels)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        parallels = [P(observations) for P in self.parallels]
        return th.cat(parallels, dim=1)



# class BaseMesh(nn.Module):
#     def __init__(self, obs_space, base_space,out_dim: int = 32):
#         super(BaseMesh,self).__init__()
#
#         obs_width = obs_space.shape[-1]
#         base_width = base_space.shape[-1]
#
#         # seq_len = obs_space.shape[0]
#         # seq_width = obs_space.shape[1]
#         # self.cnn = CNN(seq_len, seq_width)
#
#         all_width = obs_width+base_width
#         self.dense = nn.Sequential(
#             nn.Linear(all_width, obs_width),
#             nn.BatchNorm1d(obs_width),
#             nn.Mish()
#         )
#
#         self.crnn = SeqCRNN(obs_space, out_dim=out_dim)
#         self.features_dim = out_dim
#
#     def forward(self, obs: th.Tensor, base: th.Tensor) -> th.Tensor:
#         unstack = th.unbind(obs, dim=1)
#         out = [self.dense(th.concat( (row, base), dim=1)) for row in unstack]
#         stack = th.stack(out)
#         stack = stack.transpose(0,1)
#         return self.crnn(stack)

class BaseMesh(nn.Module):
    def __init__(self, obs_space, base_space,out_dim: int = 64):
        super(BaseMesh,self).__init__()

        base_width = base_space.shape[-1]

        seq_len = obs_space.shape[0]
        seq_width = obs_space.shape[1]
        self.cnn = CNN(seq_len, seq_width)

        self.base_seq = 4
        mid_out = out_dim*2
        self.rnn1 = SeqLstm(self.cnn.seq, self.cnn.feature_concat_dim, layer_num=1, out_seq=self.base_seq, out_dim=mid_out)
        self.unflat = nn.Unflatten(-1, (self.base_seq, base_width))
        self.rnn2 = SeqLstm(self.base_seq, mid_out+base_width, layer_num=1, out_seq=1, out_dim=out_dim)
        self.features_dim = out_dim


    def forward(self, obs: th.Tensor, base: th.Tensor) -> th.Tensor:
        cc = self.rnn1(self.cnn(obs))
        expand = th.cat(self.base_seq*[base], dim=1)
        expand = self.unflat(expand)
        cc = th.concat([cc, expand],dim=-1)
        return self.rnn2(cc)


class CombinedModel(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict):
        super (CombinedModel, self).__init__ ()
        self.obs = BaseMesh (observation_space.spaces[OBS], observation_space.spaces[BASE], out_dim=128)

        seq = observation_space.spaces[OBS].shape[0]
        width = observation_space.spaces[OBS].shape[1]
        # self.obs = SeqLstm( seq, width, out_dim=64)
        self.base = BaseFeature(observation_space.spaces[BASE], out_dim=8)

        self.features_dim = self.obs.features_dim
        # self.features_dim = self.obs.features_dim + self.base.features_dim
    def forward(self, observations: TensorDict) -> th.Tensor:
        obs = self.obs(observations[OBS], observations[BASE])
        return obs
        # obs = self.obs(observations[OBS])
        # base = self.base(observations[BASE])
        # return th.cat ([obs, base], dim=1)



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
