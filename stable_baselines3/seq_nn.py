
import gym
import numpy as np
import torch as th
from torch import nn
from torchinfo import summary

from stable_baselines3.common.torch_layer_base import BaseFeaturesExtractor





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


class CNN1(nn.Module):

    def __init__(self, seq_len, seq_width,  channels=[3,5], span=2):
        super (CNN1, self).__init__ ()

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
            , nn.BatchNorm2d (outch)
            , nn.Mish()
            # , nn.MaxPool2d(kernel_size=(span, 1), stride=1)

            , nn.Dropout(0.2)]

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn(self.input(observations))




# class LstmLast(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super (LstmLast, self).__init__ ()
#         # self.lstm = nn.LSTM (input_size=input_size, num_layers=2, dropout=0.1, hidden_size=hidden_size, batch_first=True)
#         self.lstm = nn.LSTM (input_size=input_size, hidden_size=hidden_size, batch_first=True)
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         return out[:,-1]




class LstmLast(nn.Module):
    def __init__(self, input_size, hidden_size, last_seq):
        super (LstmLast, self).__init__ ()
        # self.lstm = nn.LSTM (input_size=input_size, num_layers=2, dropout=0.1, hidden_size=hidden_size, batch_first=True)
        self.lstm = nn.LSTM (input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.last_seq = last_seq
    def forward(self, x):
        out, _ = self.lstm(x)
        if self.last_seq is None : return out
        else: return out[:,-self.last_seq:]


class LstmOut(nn.Module):
    def __init__(self, input_size, hidden_size):
        super (LstmOut, self).__init__ ()
        self.lstm = nn.LSTM (input_size=input_size, hidden_size=hidden_size, batch_first=True)
    def forward(self, x):
        out, _ = self.lstm(x)
        return out

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



RES =True

class ResNet(nn.Module):
    def __init__(self, inch, outch, span=3):
        super(ResNet, self).__init__()
        inter = int(inch/3)
        self.reduce_seq = span-1
        self.conv = nn.Sequential(
            nn.Conv2d (inch, inter, kernel_size=1, stride=1),
            nn.BatchNorm2d (inter),
            nn.Mish (),
            # nn.ZeroPad2d ((0, 0, span-1, 0)),
            nn.Conv2d (inter, inter, kernel_size=(span, 1), stride=1),
            nn.BatchNorm2d (inter),
            nn.Mish (),
            nn.Conv2d (inter, outch, kernel_size=1, stride=1),
            nn.BatchNorm2d (outch)
        )

        self.maxpool = nn.Sequential(
            nn.MaxPool2d (kernel_size=(span, 1), stride=1),
            # nn.ZeroPad2d ((0, 0, span - 1, 0)),
            nn.Conv2d (inch, outch, kernel_size=1, stride=1),
            nn.BatchNorm2d (outch))


        if inch==outch: self.shortcut = nn.Sequential()
        else: self.shortcut = nn.Sequential(
            nn.Conv2d (inch, outch, kernel_size=1, stride=1),
            nn.BatchNorm2d (outch))

        self.act = nn.Mish()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        conv = self.conv(observations)
        maxpool =self.maxpool(observations)
        shortcut = self.shortcut(observations)[:,:,self.reduce_seq:]
        return self.act(conv+maxpool+shortcut)




class CNN(nn.Module):

    def __init__(self, seq_len, seq_width,  init_ch=4, out_ch=16, span=2):
        super (CNN, self).__init__ ()

        network = [nn.Unflatten(-2, (1, seq_len)),
            nn.Conv2d (1, init_ch, kernel_size=(span, 1), stride=1),
            nn.BatchNorm2d (init_ch),
            nn.Mish(), nn.Dropout(0.2)]
        res_mulitple = [1,2,4,6,6]

        res_inch = init_ch
        req_reduce_sum = span -1
        for res in res_mulitple:
            res_outch = init_ch* res
            resnet = ResNet(res_inch, res_outch, span)
            req_reduce_sum += resnet.reduce_seq
            network.append(resnet)
            network.append(nn.Dropout(0.2))
            res_inch = res_outch


        out_layer = [nn.Conv2d(res_outch, out_ch, kernel_size=1, stride=1),
                     nn.BatchNorm2d(out_ch),
                     nn.Mish()]
        network.extend(out_layer)
        self.network = nn.Sequential(*network)
        self.feature_concat_dim = out_ch * seq_width
        self.seq = seq_len - req_reduce_sum


        self.vstack = nn.Flatten(start_dim=-2)
        summary(self, (1, seq_len, seq_width))
    def forward(self, observations: th.Tensor) -> th.Tensor:
        cc = self.network(observations)
        cc = th.transpose(cc, dim0=-3, dim1=-2)
        return self.vstack(cc)

class SeqCRNN (BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, out_dim: int = 64):
        super (SeqCRNN, self).__init__ (observation_space, features_dim=out_dim)

        seq_len = observation_space.shape[0]
        seq_width = observation_space.shape[1]
        self.cnn = CNN(seq_len, seq_width)

        input_size = self.cnn.feature_concat_dim
        out_seq = self.cnn.seq

        self.rnn = SeqLstm(out_seq, input_size, out_dim)


    def forward(self, observations: th.Tensor) -> th.Tensor:
        cc = self.cnn(observations)
        return self.rnn(cc)


class SeqLstm (nn.Module):
    def __init__(self, seq, input_dim, layer_num=3, out_seq=1, out_dim: int = 64):
        super (SeqLstm, self).__init__ ( )
        hiddens = np.linspace(out_dim, min(256,input_dim), layer_num, endpoint=False).astype(int).tolist()
        hiddens.reverse()
        seq_reduction = np.linspace(out_seq, seq, layer_num, endpoint=False).astype(int).tolist()
        seq_reduction.reverse()
        print(hiddens)
        print(seq_reduction)

        layers = self.inter_lstm (input_dim, hiddens, seq_reduction)
        if out_seq==1: layers.append(nn.Flatten())
        self.lstm = nn.Sequential (*layers)

        self.features_dim = out_dim
        summary(self, (1, seq, input_dim))

    def inter_lstm(self, input_size, hiddens, seq_reduction):
        layers = []
        for h, s in zip(hiddens, seq_reduction):
            layers.append(nn.Dropout (0.4))
            layers.append (LstmLast (input_size=input_size, hidden_size=h, last_seq = s))
            input_size = h



        return layers


    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.lstm(observations)

