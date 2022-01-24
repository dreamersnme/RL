from copy import deepcopy

import torch as th
from torch.utils.data import Dataset
import numpy as np

from CONFIG import *
from SOL import extractor
from stable_baselines3.common import utils
from stable_baselines3.common.running_mean_std import RunningMeanStd


class DLoader(Dataset):
    epsilon: float = 1e-8
    thresold = 0.08

    def __init__(self, data, normalizer , seq = 40, ta_seq = 30):
        self.data = data
        self.day_cnt = len(data)
        self.neg_direct = 0
        self.pos_direct = 0

        self.seq = seq
        self.ta_seq = ta_seq
        ref = self.data[0]
        self.feature_len = ref[OBS].shape[1]
        self.ta_len = ref[TA].shape[1]
        self.base_len = ref[BASE].shape[0]
        self.price_len = ref[PRICE].shape[1]

        self.daily_size = [day[OBS].shape[0] - seq+1 for day in self.data]
        self.daily_idx = np.array([0]+self.daily_size).cumsum()
        self.normalizer = normalizer


    def day_search(self, idx):
        left = 0
        right = self.day_cnt
        while right-left >1:
            mid = int((left + right)/2)
            if idx >= self.daily_idx[mid]: left = mid
            else: right = mid
        idx = idx - self.daily_idx[left]
        return left, idx

    def normalize(self, key, obs):
        obs_rms = self.normalizer[key]
        return obs_rms.normalize(obs)

    def denorm_target(self, target):
        obs_rms = self.normalizer[PRICE]
        return obs_rms.denormalize(target)


    def abs_diff(self, pred, true):
        return np.abs(self.denorm_target(pred)- self.denorm_target(true))

    def __getitem__(self, idx):
        if idx >= self.daily_idx[-1]: raise IndexError(idx)
        day_no, idx = self.day_search(idx)
        day = self.data[day_no]
        s_idx = idx
        e_idx = idx+self.seq-1
        obs = day[OBS][s_idx: e_idx+1]
        ta = day[TA][e_idx -self.ta_seq +1 :e_idx+1]
        return ({OBS:obs, TA:ta, BASE:day[BASE]}, day[PRICE][e_idx], day[DIRECT][e_idx])


    def __len__(self):
        return self.daily_idx[-1]

