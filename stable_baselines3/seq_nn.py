
import gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layer_base import BaseFeaturesExtractor


class BaseFeature(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, out_dim=16):
        super(BaseFeature, self).__init__(observation_space, features_dim=out_dim)
        in_dim = gym.spaces.utils.flatdim(observation_space)
        self.dense = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.dense(observations)



class SeqFeature(BaseFeaturesExtractor):
    def __init__(self, obs_space: gym.spaces.Box):
        super (SeqFeature, self).__init__ (obs_space, features_dim=1)
        extractors =[ SeqCRNN(obs_space, 64)]
        total_concat_size = 0
        for ex in extractors:
            total_concat_size += ex._features_dim
        self._features_dim = total_concat_size

        self.extractors = nn.ModuleList (extractors)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        encoded_tensor_list = []
        for ext in self.extractors:
            encoded_tensor_list.append(ext(obs))
        return th.cat(encoded_tensor_list, dim=1)

class SeqCNN (BaseFeaturesExtractor):
    ch1 = 7
    ch2 = 3
    def __init__(self, observation_space: gym.spaces.Box, span=3, outdim: int = 64):
        super (SeqCNN, self).__init__ (observation_space, features_dim=outdim)
        seq_len = observation_space.shape[0]
        seq_width = observation_space.shape[1]
        span = min (seq_len, span)
        n_cnn = (seq_len - span +1) * seq_width * self.ch1

        self.cnn = nn.Sequential (
            nn.Unflatten(-2, (1,seq_len)),
            nn.Conv2d (1, self.ch1, kernel_size=(span, 1), stride=1),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Conv2d(self.ch1, self.ch2, kernel_size=(span, 1), stride=1),
            nn.Mish(),
            nn.Flatten (),
            nn.Linear (n_cnn, outdim),
            nn.Mish ()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn(observations)



class SeqLast(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        super (SeqLast, self).__init__ (observation_space, observation_space.shape[-1])
        self.flatten = nn.Flatten ()
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten (observations[:,-1])

class LstmLast(nn.Module):
    def __init__(self, seq_width, hidden):
        super (LstmLast, self).__init__ ()
        self.lstm = nn.LSTM (input_size=seq_width, num_layers=2, dropout=0.2, hidden_size=hidden, batch_first=True)
        # self.lstm = nn.LSTM (input_size=seq_width, hidden_size=hidden, batch_first=True)
    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:,-1]



class SeqCRNN (BaseFeaturesExtractor):
    ch1 = 3
    ch2= 5
    def __init__(self, observation_space: gym.spaces.Box, out_dim: int = 64, span=3):
        super (SeqCRNN, self).__init__ (observation_space, features_dim=out_dim)
        seq_len = observation_space.shape[0]
        seq_width = observation_space.shape[1]
        span = min (seq_len, span)
        out_len = (seq_len - span +1)
        assert out_len > 1

        self.crnn = nn.Sequential (
            nn.Unflatten (-2, (1, seq_len)),
            nn.Conv2d (1, self.ch1, kernel_size=(span, 1), bias=False),
            nn.BatchNorm2d(self.ch1),
            nn.Dropout(0.2),
            nn.Mish (),
            nn.Conv2d(self.ch1, self.ch2, kernel_size=(span, 1), bias=False),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU(),
            nn.Conv2d (self.ch2, 1, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(1),
            nn.Dropout(0.2),
            nn.Flatten (end_dim=-2),
            LstmLast (seq_width=seq_width, hidden=out_dim),
            nn.Dropout(0.2),
            nn.ReLU ()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:

        return self.crnn(observations)


class SeqLstm (BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, out_dim: int = 64):
        super (SeqLstm, self).__init__ (observation_space, features_dim=out_dim)
        seq_width = observation_space.shape[1]
        self.lstm = nn.Sequential(
            LstmLast (seq_width=seq_width, hidden=out_dim),
            nn.Mish(),
            nn.Dropout(0.2)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.lstm(observations)



