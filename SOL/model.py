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


class OutterModel(nn.Module):

    def __init__(self, feature_length, seq, ta_len, ta_seq, base_len):
        obs = spaces.Box(low=-np.inf, high=np.inf, shape=(seq, feature_length))
        ta = spaces.Box(low=-np.inf, high=np.inf, shape=(ta_seq, ta_len))
        base = spaces.Box(low=-np.inf, high=np.inf, shape=(base_len,))
        observation_space = spaces.Dict(OrderedDict([(OBS, obs), (TA, ta), (BASE, base)]))

        super(OutterModel, self).__init__()
        from stable_baselines3.common.torch_layers import CombinedExtractor
        self.module = CombinedExtractor(observation_space)
        dim1 = self.module.features_dim
        dim2= int(dim1/2)

        self.agg = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.BatchNorm1d(dim2),
            nn.Mish(),
            nn.Dropout(0.2)
        )
        self.price = nn.Linear (dim2, 5)
        self.direction =  nn.Sequential(
            nn.Linear(dim2, 5),
            nn.Tanh())

    def forward(self, observations: TensorDict) -> th.Tensor:
        feature = self.module(observations)
        agg = self.agg(feature)
        return  th.cat([self.price(agg), self.direction(agg)], dim=1)
