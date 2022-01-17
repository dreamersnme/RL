from collections import OrderedDict

import torch.nn as nn
import torch as th
import numpy as np
from gym import spaces

from CONFIG import *
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.seq_nn import SeqFeature, BaseFeature

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()
    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = th.sqrt(criterion(x, y))
        return loss

class CombinedModel(nn.Module):
    def __init__(self, feature_length, seq, base_len):

        super(CombinedModel, self).__init__()
        obs = spaces.Box(low=-np.inf, high=np.inf, shape=(seq, feature_length))
        base = spaces.Box(low=-np.inf, high=np.inf, shape=(base_len,))
        self.observation_space = spaces.Dict(OrderedDict([(OBS, obs), (BASE, base)]))

        extractors = {OBS: SeqFeature(self.observation_space.spaces[OBS])
            , BASE: BaseFeature(self.observation_space[BASE])}

        total_concat_size = sum([module.features_dim for module in extractors.values()])
        self.extractors = nn.ModuleDict(extractors)
        self.features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)

class OutterModel(nn.Module):
    def __init__(self, feature_length, seq, base_len):
        super(OutterModel, self).__init__()
        self.module = CombinedModel(feature_length, seq, base_len)
        dim1 = self.module.features_dim
        dim2= int(dim1/2)
        self.agg = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Linear(dim2, dim2),
            nn.ReLU(),
            nn.Linear(dim2, 1),
        )
    def forward(self, observations: TensorDict) -> th.Tensor:
        return self.agg(self.module(observations))
