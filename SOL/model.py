from abc import abstractmethod
from collections import OrderedDict

import gym
import torch.nn as nn
import torch as th
import numpy as np
from gym import spaces
from talib import abstract

from CONFIG import *
from SOL.DLoader import TA
from stable_baselines3.common.torch_layer_base import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.seq_nn import SeqFeature, BaseFeature, SeqLstm, SeqCRNN, SeqCNN

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

class TaNN(ObsNN):
    pass


class CombinedModel(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict):
        super (CombinedModel, self).__init__ (observation_space, features_dim=1)
        extractors = self.get_dict(observation_space)
        total_concat_size = sum ([module.features_dim for module in extractors.values()])
        self.extractors = nn.ModuleDict (extractors)
        self._features_dim = total_concat_size

    def get_dict(self, observation_space):
        return {OBS: ObsNN (observation_space.spaces[OBS])
            , TA: TaNN (observation_space.spaces[TA])
            , BASE: BaseFeature (observation_space[BASE], out_dim=4)}

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items ():
            encoded_tensor_list.append (extractor (observations[key]))
        return th.cat (encoded_tensor_list, dim=1)



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
            nn.Dropout(0.2)
        )
        self.price = nn.Linear (dim2, spec.price_len)
        self.direction =  nn.Sequential(
            nn.Linear(dim2, spec.price_len),
            nn.Tanh())

    def forward(self, observations: TensorDict) -> th.Tensor:
        feature = self.module(observations)
        agg = self.agg(feature)
        return  th.cat([self.price(agg), self.direction(agg)], dim=1)
