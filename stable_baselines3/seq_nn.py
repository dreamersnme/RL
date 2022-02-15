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




class Incept(nn.Module):
    def __init__(self, inch, outch):
        super(Incept, self).__init__()
        layer = 5
        inter_out = outch
        each = int(inter_out / layer)
        # each_in = int(each / 2)
        # each_in = each
        each_in = inch
        denseout = inter_out - each * (layer - 1)

        self.dense = nn.Sequential(
            # nn.BatchNorm1d(inch),
            nn.Conv1d(inch, denseout, kernel_size=1, stride=1),
        )

        self.conv2 = self.conv_module(2, inch, each_in, each)
        self.conv3 = self.conv_module(3, inch, each_in, each)
        self.conv5 = self.conv_module(5, inch, each_in, each)

        self.maxpool = nn.Sequential(
            # nn.BatchNorm1d(inch),
            nn.MaxPool1d(kernel_size=5, stride=1),
            # nn.Mish(),
            # nn.BatchNorm1d(inch),
            nn.Conv1d(inch, each, kernel_size=1, stride=1),
            # nn.BatchNorm1d(each)
        )
        # self.minpool = nn.Sequential(
        #     nn.MaxPool1d(kernel_size=4, stride=1),
        #     nn.BatchNorm1d(inch), nn.Mish(),
        #     nn.Conv1d(inch, each, kernel_size=1, stride=1),
        #     nn.BatchNorm1d(each)
        # )
        # self.act =nn.Sequential( nn.Mish(), nn.BatchNorm1d(outch))
        self.act = nn.Mish()
        self.reduce_seq = 4

    def conv_module(self, span, inch, each_in, each):
        return nn.Sequential(
            # nn.BatchNorm1d(inch),
            # nn.Conv1d(inch, each_in, kernel_size=1, stride=1),
            # nn.Mish(),
            # nn.BatchNorm1d(each_in),
            nn.Conv1d(each_in, each, kernel_size=span, stride=1),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        dense = self.dense(observations[:, :, self.reduce_seq:])
        conv2 = self.conv2(observations[:, :, self.reduce_seq - 1:])
        conv3 = self.conv3(observations[:, :, self.reduce_seq - 2:])
        conv5 = self.conv5(observations)
        maxpool = self.maxpool(observations)
        return self.act(th.concat([dense, conv2, conv3, conv5, maxpool], dim=1))


class SeqCNN(nn.Module):

    def __init__(self, seq_len, init_ch, out_dim):
        super(SeqCNN, self).__init__()
        network = []
        init_ch = init_ch*2
        # res_mulitple = [10, 20, 10, 10]

        res_mulitple = [1, 1, 1, 1]

        inter_dim = int(init_ch/0.5)
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
