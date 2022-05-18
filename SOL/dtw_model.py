



from CONFIG import *
from SOL.soft_dtw_cuda import SoftDTW
from SOL.trann import TARNN
from stable_baselines3.common import utils

from stable_baselines3.common.type_aliases import TensorDict
import gym

import numpy as np
import torch as th
from torch import nn
from torchinfo import summary
import math
from torch.nn import Parameter, init
from torch import Tensor



class DtwLayer(nn.Module):
    def __init__(self, align_dim: int, out_dim: int,  bandwidth=0.5, seq_weight = True, bias: bool = True,
                 device=None, dtype=None) -> None:
        super (DtwLayer, self).__init__ ()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.dtw_loss = SoftDTW (use_cuda=True, gamma=0.2, bandwidth=bandwidth)
        self.out_dim = out_dim
        self.align_seq = Parameter(th.empty((out_dim, align_dim), **factory_kwargs))

        self.bias = Parameter(th.empty((out_dim,1), **factory_kwargs)) if bias else None
        self.weight = Parameter (th.empty ((out_dim, align_dim), **factory_kwargs)) if seq_weight else None
        self.reset_parameters()

        # self.weight.register_hook (lambda grad: print ("===",grad))

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.align_seq, a=math.sqrt(5))
        if self.weight is not None:
            # nn.init.xavier_uniform_(self.weight)
            init.constant_ (self.weight, 1.0)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.align_seq)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if self.weight is None:
            return self.dtw_loss(input, self.align_seq + self.bias, None)
        else:
            return self.dtw_loss(input, self.align_seq + self.bias, self.weight.exp())

    def extra_repr(self) -> str:
        return 'out_dim={}, bias={}'.format(
            self.out_dim, self.bias is not None
        )

class SpanCutter(nn.Module):
    def __init__(self, maxspan, span):
        super(SpanCutter, self).__init__()
        self.start = maxspan - span

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return observations[:, :, self.start:]


class Incept(nn.Module):
    def __init__(self, inch, outch, convspan, poolspan):
        super(Incept, self).__init__()
        self.max_span = max(convspan + poolspan)


        layer = len(convspan)+len(poolspan) +1
        inter_out = outch
        each = int(inter_out / layer)
        # each_in = int(each / 2)
        # each_in = each
        each_in = inch
        denseout = inter_out - each * (layer - 1)

        dense = nn.Sequential(
            SpanCutter(self.max_span , 1),
            nn.Conv1d(inch, denseout, kernel_size=1, stride=1),
        )

        convs=[self.conv_module(ii, inch, each_in, each) for ii in convspan]


        pools = [self.max_module(ii, inch, each) for ii in poolspan]

        self.layers = nn.ModuleList([dense] + convs + pools)

        self.act = nn.Mish()
        self.reduce_seq = self.max_span -1

    def conv_module(self, span, inch, each_in, each):
        return nn.Sequential(
            SpanCutter(self.max_span , span),
            nn.Conv1d(each_in, each, kernel_size=span, stride=1),
        )
    def max_module(self, span, inch, each):
        return nn.Sequential(
            SpanCutter(self.max_span , span),
            nn.MaxPool1d(kernel_size=span, stride=1),
            nn.Conv1d(inch, each, kernel_size=1, stride=1),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        out = [layer(observations) for layer in self.layers]
        out = th.concat(out, dim=1)
        return self.act(out)




class LstmLast(nn.Module):
    def __init__(self, input_size, hidden_size, last_seq =None):
        super(LstmLast, self).__init__()
        # self.lstm = nn.LSTM (input_size=input_size, num_layers=2, dropout=0.1, hidden_size=hidden_size, batch_first=True)
        self.lstm = TARNN(input_size=input_size, hidden_size=hidden_size, n_timesteps=20)
        self.last_seq = last_seq

    def forward(self, x):
        out, _ = self.lstm(x)
        if self.last_seq is None:
            return out
        else:
            return out[:, -self.last_seq:]


class SeqCNN(nn.Module):

    def __init__(self, seq_len, init_ch, out_dim):
        super(SeqCNN, self).__init__()


        inter_dim = int(init_ch * 2)
        redu_dim =  30# init_ch
        redu_dim2 = 40

        network = [nn.Conv1d(init_ch, inter_dim, kernel_size=1),nn.Mish()]
        incpt = [Incept (inter_dim, inter_dim, [2,3], [2])]
        self.seq = seq_len - sum([ ly.reduce_seq for ly in incpt])
        reduce = [nn.Conv1d (inter_dim, redu_dim, kernel_size=1), nn.Mish ()]
        network.extend(incpt)
        network.extend(reduce)


        self.network = nn.Sequential (*network)

        self.dtw = LstmLast(redu_dim, redu_dim2)

        inter_dim = int ( (init_ch+out_dim) /2 )
        self.out = nn.Sequential(
            # nn.Linear(init_ch, inter_dim), nn.Mish(),
            # nn.Linear (inter_dim, out_dim),
            nn.Linear (redu_dim2, out_dim),
            nn.Tanh())

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # obs = th.concat( (observations, -observations), axis = -1)
        print(obs.size())
        obs = obs.transpose(-2, -1)
        print(obs.size())
        xx = self.network(obs)
        print(xx.size())
        xx = xx.transpose(-2, -1).squeeze(axis=1)
        print(xx.size())
        xx = self.dtw(xx)
        return self.out(xx)







class BaseMesh(nn.Module):
    def __init__(self, obs_space,out_dim: int = 64):
        super(BaseMesh,self).__init__()
        seq_len = obs_space.shape[0]
        seq_width = obs_space.shape[1]
        self.cnn = SeqCNN(seq_len, seq_width, out_dim)
        self.features_dim = out_dim
        self.testinput=(1, seq_len, seq_width)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.cnn(obs)



class CombinedModel(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, out_dim):
        super (CombinedModel, self).__init__ ()
        self.obs = BaseMesh (observation_space.spaces[OBS], out_dim)
        self.features_dim = self.obs.features_dim
        # self.features_dim = self.obs.features_dim + self.base.features_dim
    def forward(self, observations: TensorDict) -> th.Tensor:
        obs = self.obs(observations[OBS])
        return obs

    def printsummary(self):
        summary(self.obs, self.obs.testinput)

    def predict(self,  observations: TensorDict) -> th.Tensor:
        tensor = utils.obs_as_tensor(observations, DEVICE)
        with th.no_grad():
            self.eval()
            pred = self.forward(tensor)
            self.train()
        return pred.cpu().numpy()



class OutterModel2(nn.Module):
    def __init__(self, spec):
        observation_space = spec.data_space
        super(OutterModel2, self).__init__()
        self.module = CombinedModel(observation_space, spec.featured_feature)
        dim1 = self.module.features_dim

        self.price = nn.Linear (dim1, spec.price_len)
        self.direction =  nn.Sequential(
            nn.Linear(dim1, spec.price_len),
            nn.Tanh())
        # self.module.requires_grad_ (False)

    @property
    def feature_ext(self): return self.module
    def printsummary(self):
        self.module.printsummary()

    def forward(self, observations: TensorDict) -> th.Tensor:
        agg = self.module(observations)
        return th.cat([self.price(agg), self.direction(agg)], dim=1)

    def predict_encoded(self,  encoded) -> th.Tensor:
        tensor = utils.obs_as_tensor(encoded, DEVICE)
        with th.no_grad():
            self.eval()
            pred = self.price(tensor)
            self.train()
        return pred.cpu().numpy()

    def predict(self,  observations: dict) -> th.Tensor:
        tensor = utils.obs_as_tensor(observations, DEVICE)
        with th.no_grad():
            self.eval()
            pred = self.forward(tensor)
            self.train()
        return pred.cpu().numpy()
