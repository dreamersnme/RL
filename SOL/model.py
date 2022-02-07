

from CONFIG import *

from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.seq_nn import *

class BaseFeature(nn.Module):
    def __init__(self, observation_space: gym.Space, out_dim=16):
        super(BaseFeature, self).__init__()
        in_dim = gym.spaces.utils.flatdim(observation_space)
        self.dense = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Mish()
        )
        self.features_dim=out_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.dense(observations)




class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()
    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = th.sqrt(criterion(x, y))
        return loss




class BaseMesh(nn.Module):
    def __init__(self, obs_space,out_dim: int = 64):
        super(BaseMesh,self).__init__()
        seq_len = obs_space.shape[0]
        seq_width = obs_space.shape[1]
        self.cnn = SeqCNN(seq_len, seq_width, out_dim)
        self.features_dim = out_dim
        summary (self, (1, seq_len, seq_width))
    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.cnn(obs)


class CombinedModel(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict):
        super (CombinedModel, self).__init__ ()
        self.obs = BaseMesh (observation_space.spaces[OBS])
        self.features_dim = self.obs.features_dim
        # self.features_dim = self.obs.features_dim + self.base.features_dim
    def forward(self, observations: TensorDict) -> th.Tensor:
        obs = self.obs(observations[OBS])
        return obs





class OutterModel(nn.Module):
    def __init__(self, spec):
        observation_space = spec.data_space
        super(OutterModel, self).__init__()
        self.module = CombinedModel(observation_space)
        dim1 = self.module.features_dim

        self.price = nn.Linear (dim1, spec.price_len)
        self.direction =  nn.Sequential(
            nn.Linear(dim1, spec.price_len),
            nn.Tanh())
        # self.module.requires_grad_ (False)


    def forward(self, observations: TensorDict) -> th.Tensor:
        agg = self.module(observations)

        return th.cat([self.price(agg), self.direction(agg)], dim=1)
