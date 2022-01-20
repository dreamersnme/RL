from collections import OrderedDict

import torch.nn as nn
import torch as th
import numpy as np
from gym import spaces

from CONFIG import *
from SOL.DLoader import TA
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
        outdim = 16

        self.features_dim = outdim *3
        self.lstm = SeqLstm(space, out_dim=outdim)
        self.crnn = SeqCRNN(space, out_dim=outdim)
        self.cnn = SeqCNN(space, out_dim=outdim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return th.cat([self.lstm(observations),self.cnn(observations), self.crnn(observations)], dim=1)


class CombinedModel(nn.Module):
    def __init__(self, feature_length, seq, ta_len, ta_seq, base_len):

        super(CombinedModel, self).__init__()
        obs = spaces.Box(low=-np.inf, high=np.inf, shape=(seq, feature_length))
        ta = spaces.Box(low=-np.inf, high=np.inf, shape=(ta_seq, ta_len))
        base = spaces.Box(low=-np.inf, high=np.inf, shape=(base_len,))
        self.observation_space = spaces.Dict(OrderedDict([(OBS, obs), (TA, ta),(BASE, base)]))

        extractors = {OBS: ObsNN(self.observation_space.spaces[OBS])
            , TA: ObsNN(self.observation_space.spaces[TA])
            , BASE: BaseFeature(self.observation_space[BASE], out_dim=4)}

        total_concat_size = sum([module.features_dim for module in extractors.values()])
        self.extractors = nn.ModuleDict(extractors)
        self.features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)

class OutterModel(nn.Module):
    def __init__(self, feature_length, seq, ta_len, ta_seq, base_len):
        super(OutterModel, self).__init__()
        self.module = CombinedModel( feature_length, seq, ta_len, ta_seq, base_len)
        dim1 = self.module.features_dim
        dim2= int(dim1/2)

        self.agg = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.BatchNorm1d(dim2),
            nn.Mish()
        )
        self.price = nn.Linear (dim2, 3)
        self.direction =  nn.Sequential(
            nn.Linear(dim2, 2),
            nn.Tanh())

    def forward(self, observations: TensorDict) -> th.Tensor:
        feature = self.module(observations)
        return  th.cat([self.price(self.agg(feature)), self.direction(self.agg(feature))], dim=1)
