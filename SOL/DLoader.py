from copy import deepcopy

import torch as th
from torch.utils.data import Dataset
import numpy as np

from CONFIG import *
from SOL import extractor
from stable_baselines3.common.running_mean_std import RunningMeanStd

TARGET ="target"
TA ='ta'
class DLoader(Dataset):
    epsilon: float = 1e-8
    thresold = 0.06

    def __init__(self, data, normalizer = None, seq = 15, ta_seq = 10):
        self.data = data
        self.day_cnt = len(data)
        self.neg_direct = 0
        self.pos_direct = 0

        self.target2 = [self.direction(d.price) for d in data]
        self.target = [d.price.reshape(-1,1) for d in data]
        self.seq = seq
        self.ta_seq = ta_seq
        self.feature_len = self.data[0].data.shape[1]
        self.ta_len = self.data[0].ta.shape[1]
        self.base_len = self.data[0].base.shape[0]
        self.daily_size = [day.data.shape[0] - seq+1 for day in self.data]
        self.daily_idx = np.array([0]+self.daily_size).cumsum()
        self.normalizer = normalizer
        if normalizer is None: self.normalizer = self._get_normalizer()
        else: self._get_normalizer()


    def direction(self, price):
        price
        surge = []
        for p in price:


            if p <= -self.thresold: index = -1; self.pos_direct += 1
            elif p>=self.thresold: index = 1; self.neg_direct +=1
            else: index = 0
            surge.append([index])

        return np.array(surge)

    def _get_normalizer(self):
        obsN = RunningMeanStd(shape=(self.seq, self.feature_len))
        taN = RunningMeanStd(shape=(self.ta_seq, self.ta_len))
        baseN = RunningMeanStd(shape=(self.base_len,))
        targetN = RunningMeanStd(shape=(1,))


        for idx in range(self.__len__()):
            obs, ta, base, target,direct = self.__getitem(idx)
            obsN.update(obs)
            taN.update(ta)
            baseN.update(base)
            targetN.update(target)


        return {OBS: obsN, TA:taN, BASE:baseN, TARGET:targetN}

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
        return ((obs - obs_rms.mean) / np.sqrt(obs_rms.var + self.epsilon)).astype(np.float32)

    def denorm_target(self, target):
        obs_rms = self.normalizer[TARGET]
        return (target * np.sqrt (obs_rms.var + self.epsilon)) + obs_rms.mean


    def abs_diff(self, pred, true):
        return np.abs(self.denorm_target(pred)- self.denorm_target(true))

    def __getitem(self, idx):
        if idx >= self.daily_idx[-1]: raise IndexError(idx)
        day_no, idx = self.day_search(idx)
        day = self.data[day_no]
        s_idx = idx
        e_idx = idx+self.seq-1
        obs = day.data[s_idx: e_idx+1]
        ta = day.ta[e_idx -self.ta_seq +1 :e_idx+1]
        return obs, ta, day.base, self.target[day_no][e_idx], self.target2[day_no][e_idx]

    def __getitem__(self, idx):
        obs, ta, base, target, target2 = self.__getitem(idx)
        return ({OBS:self.normalize(OBS,obs)
                    ,TA:self.normalize(TA, ta)
                    ,BASE:self.normalize(BASE,base)}
                , self.normalize(TARGET,target), target2)

    def __len__(self):
        return self.daily_idx[-1]


if __name__ == "__main__":
    data, test = extractor.load_ml()

    loader = DLoader(data)
    print(loader.__len__())
    # print (loader.__getitem__ (11352))

    # print(loader.day_search(0))
    # print (loader.day_search (593))
    # print (loader.day_search (594))
    # print (loader.day_search (1188))
    # print (loader.day_search (8900))
    # print (loader.day_search (8901))



