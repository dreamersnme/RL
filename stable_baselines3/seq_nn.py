import gym
import numpy as np
import torch as th
from torch import nn
from torchinfo import summary

from stable_baselines3.common.torch_layer_base import BaseFeaturesExtractor


class LstmLast(nn.Module):
    def __init__(self, input_size, hidden_size, last_seq):
        super(LstmLast, self).__init__()
        # self.lstm = nn.LSTM (input_size=input_size, num_layers=2, dropout=0.1, hidden_size=hidden_size, batch_first=True)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.last_seq = last_seq

    def forward(self, x):
        out, _ = self.lstm(x)
        if self.last_seq is None:
            return out
        else:
            return out[:, -self.last_seq:]


class SpanCutter(nn.Module):
    def __init__(self, maxspan, span):
        super(SpanCutter, self).__init__()
        self.start = maxspan - span

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return observations[:, :, self.start:]


class Incept(nn.Module):
    def __init__(self, inch, outch):
        super(Incept, self).__init__()

        convspan = [2,3,4,5]
        poolspan = [3]

        layer = len(convspan)+len(poolspan) +1
        inter_out = outch
        each = int(inter_out / layer)
        # each_in = int(each / 2)
        # each_in = each
        each_in = inch
        denseout = inter_out - each * (layer - 1)

        dense = nn.Sequential(
            SpanCutter(5, 1),
            nn.Conv1d(inch, denseout, kernel_size=1, stride=1),
        )


        convs=[self.conv_module(ii, inch, each_in, each) for ii in convspan]


        pools = [self.max_module(ii, inch, each) for ii in poolspan]

        self.layers = nn.ModuleList([dense] + convs + pools)

        self.act = nn.Mish()
        self.reduce_seq = 4

    def conv_module(self, span, inch, each_in, each):
        return nn.Sequential(
            SpanCutter(5, span),
            # nn.BatchNorm1d(inch),
            # nn.Conv1d(inch, each_in, kernel_size=1, stride=1),
            # nn.Mish(),
            # nn.BatchNorm1d(each_in),
            nn.Conv1d(each_in, each, kernel_size=span, stride=1),
        )
    def max_module(self, span, inch, each):
        return nn.Sequential(
            SpanCutter(5, span),
            nn.MaxPool1d(kernel_size=span, stride=1),
            # nn.Mish(),
            # nn.BatchNorm1d(inch),
            nn.Conv1d(inch, each, kernel_size=1, stride=1),
            # nn.BatchNorm1d(each)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        out = [layer(observations) for layer in self.layers]
        out = th.concat(out, dim=1)
        return self.act(out)


class SeqCNN(nn.Module):

    def __init__(self, seq_len, init_ch, out_dim):
        super(SeqCNN, self).__init__()
        network = []
        init_ch = init_ch*2
        # res_mulitple = [10, 20, 10, 10]

        # res_mulitple = [1, 1, 1, 1]
        res_mulitple = [2, 1]

        inter_dim = int(init_ch/0.7)
        network.extend([ nn.Conv1d(init_ch, inter_dim, kernel_size=1, stride=1),
                         nn.BatchNorm1d(inter_dim),
                         nn.Tanh(),
                         nn.Dropout(0.4)
                         ])

        init_ch = inter_dim
        res_inch = init_ch
        req_reduce_sum = 0
        for res in res_mulitple:
            res_outch = int(init_ch * res)
            resnet = Incept(res_inch, res_outch)
            req_reduce_sum += resnet.reduce_seq
            #
            network.append(resnet)
            network.append(nn.Dropout(0.4))
            res_inch = res_outch

        inter_dim = int((res_inch + out_dim) / 2)
        inter_dim = res_inch

        self.out = nn.Sequential(

            # nn.Dropout(0.4),
            # nn.Linear(res_inch, inter_dim),
            # nn.Mish(),

            nn.Dropout(0.6),
            nn.Linear(inter_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Tanh())

        self.network = nn.Sequential(*network)
        self.feature_concat_dim = res_mulitple[-1] * init_ch
        self.seq = seq_len - req_reduce_sum

    def forward(self, observations: th.Tensor) -> th.Tensor:
        obs = th.concat( (observations, -observations), axis = -1)
        obs = obs.transpose(-2, -1)
        xx = self.network(obs).transpose(-2, -1)[:, -1]
        return self.out(xx)


class SeqLstm(nn.Module):

    def __init__(self, seq, input_dim, layer_num=3, out_seq=1, out_dim: int = 64):
        super(SeqLstm, self).__init__()
        hiddens = np.linspace(out_dim, min(256, input_dim), layer_num, endpoint=False).astype(int).tolist()
        hiddens.reverse()
        seq_reduction = np.linspace(out_seq, seq, layer_num, endpoint=False).astype(int).tolist()
        seq_reduction.reverse()

        layers = self.inter_lstm(input_dim, hiddens, seq_reduction)
        if out_seq == 1: layers.append(nn.Flatten())
        self.lstm = nn.Sequential(*layers)

        self.features_dim = out_dim
        summary(self, (1, seq, input_dim))

    def inter_lstm(self, input_size, hiddens, seq_reduction):
        layers = []
        for h, s in zip(hiddens, seq_reduction):
            layers.append(nn.Dropout(0.2))
            layers.append(LstmLast(input_size=input_size, hidden_size=h, last_seq=s))
            input_size = h
        return layers

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.lstm(observations)
