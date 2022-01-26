
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
            nn.Mish()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.dense(observations)


class LstmLast(nn.Module):
    def __init__(self, seq_width, hidden):
        super (LstmLast, self).__init__ ()
        self.lstm = nn.LSTM (input_size=seq_width, num_layers=2, dropout=0.2, hidden_size=hidden, batch_first=True)
        # self.lstm = nn.LSTM (input_size=seq_width, hidden_size=hidden, batch_first=True)
    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:,-1]

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


class CNN(nn.Module):

    def __init__(self, seq_len, seq_width,  channels=[3,5], span=2):
        super (CNN, self).__init__ ()

        span = min (seq_len, span)
        layer_cnt = len(channels)
        # out_seq_len = (seq_len - layer_cnt*span*2 + layer_cnt*2)
        out_seq_len = seq_len - layer_cnt * (-span +1)

        assert out_seq_len > 1
        self.feature_concat_dim = seq_width * channels[-1]
        self.feature_dim = out_seq_len * self.feature_concat_dim
        self.input = nn.Unflatten(-2, (1, seq_len))
        cnns = []
        in_dim = 1

        for ch in channels:
            cnns.extend(self.cnn_module(in_dim, ch, span))
            in_dim = ch
        self.cnn = nn.Sequential(*cnns)

    def cnn_module(self, inch, outch, span):
        return [nn.Conv2d(inch, outch, kernel_size=(span, 1), stride=1)
            , nn.Mish()
            # , nn.MaxPool2d(kernel_size=(span, 1), stride=1)
            , nn.BatchNorm2d(outch)
            , nn.Dropout(0.2)]

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn(self.input(observations))



class SeqCNN (BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, out_dim: int = 64):
        super (SeqCNN, self).__init__ (observation_space, features_dim=out_dim)
        seq_len = observation_space.shape[0]
        seq_width = observation_space.shape[1]
        self.cnn = CNN(seq_len, seq_width)
        self.flatten = nn.Sequential (
            nn.Flatten (start_dim=1), nn.Linear (self.cnn.feature_dim, out_dim), nn.Mish ())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        ss = self.cnn(observations)
        return self.flatten(ss)

class SeqLstm (BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, out_dim: int = 64):
        super (SeqLstm, self).__init__ (observation_space, features_dim=out_dim)
        seq_width = observation_space.shape[1]
        self.lstm = nn.Sequential(
            LstmLast (seq_width=seq_width, hidden=out_dim),
            nn.Mish(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.lstm(observations)

class SeqCRNN (BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, out_dim: int = 64):
        super (SeqCRNN, self).__init__ (observation_space, features_dim=out_dim)

        seq_len = observation_space.shape[0]
        seq_width = observation_space.shape[1]
        self.cnn = CNN(seq_len, seq_width)

        self.rnn = nn.Sequential (
            nn.Flatten (start_dim=-2),
            LstmLast(seq_width=self.cnn.feature_concat_dim, hidden=out_dim),
            nn.Mish(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        cc = self.cnn(observations)
        cc = th.transpose(cc, dim0=-3, dim1=-2)
        return self.rnn(cc)


        # return self.rnn(self.cnn(observations))




