from copy import deepcopy

import torch as th
from torch.utils.data import Dataset
import numpy as np

from CONFIG import *
from SOL import extractor
from stable_baselines3.common import utils
from stable_baselines3.common.running_mean_std import RunningMeanStd

PRICE = "price"
TA ='ta'
DIRECT = "direct"
DEVICE ="cuda"
class DLoader(Dataset):
    epsilon: float = 1e-8
    thresold = 0.08

    def __init__(self, data, normalizer = None, seq = 40, ta_seq = 30):
        self.data = data
        self.day_cnt = len(data)
        self.neg_direct = 0
        self.pos_direct = 0
        self.direction = [self.cal_direction(d.price) for d in data]
        self.price = [d.price for d in data]
        self.price_len = self.price[0].shape[1]
        self.seq = seq
        self.ta_seq = ta_seq
        self.feature_len = self.data[0].data.shape[1]
        self.ta_len = self.data[0].ta.shape[1]
        self.base_len = self.data[0].base.shape[0]
        self.daily_size = [day.data.shape[0] - seq+1 for day in self.data]
        self.daily_idx = np.array([0]+self.daily_size).cumsum()
        self.normalizer = normalizer
        if normalizer is None: self.normalizer = self._get_normalizer()
        self.up_gpu()

    def cal_price(self, price):
        return price
        #
        # plus = np.array([p if p>0 else 0 for p in price])
        # miunus = np.array([ p if p < 0 else 0 for p in price])
        # prices =  np.stack((price, plus, miunus ), axis=1)
        # return prices

    def cal_direction(self, price):
        plus = np.where(price>=self.thresold, 1, 0)
        mius = np.where(price<=self.thresold, -1, 0)
        return plus + mius

        # surge = []
        # for p in price:
        #     if p>=self.thresold: index = [1,0]; self.neg_direct +=1
        #     elif p <= -self.thresold: index = [0, -1]; self.pos_direct += 1
        #     else: index = [0,0]
        #     surge.append(index)
        # return np.array(surge)

    def _get_normalizer(self):
        obsN = RunningMeanStd(shape=(self.feature_len,))
        taN = RunningMeanStd(shape=(self.ta_len,))
        baseN = RunningMeanStd(shape=(self.base_len,))
        priceN = RunningMeanStd(shape=(self.price_len,))
        for day_no in range(self.day_cnt):
            day = self.data[day_no]
            obsN.update(day.data)
            taN.update(day.ta)
            baseN.update(day.base)
            priceN.update(self.price[day_no])

        return {OBS: obsN, TA:taN, BASE:baseN, PRICE:priceN}

    def up_gpu(self):
        days =dict()
        for day_no in range(self.day_cnt):
            day = self.data[day_no]
            obs = utils.obs_as_tensor(self.normalize(OBS, day.data), DEVICE)
            ta = utils.obs_as_tensor(self.normalize(TA, day.ta), DEVICE)
            base = utils.obs_as_tensor(self.normalize(BASE, day.base), DEVICE)
            price = utils.obs_as_tensor(self.normalize(PRICE, self.price[day_no]), DEVICE)
            direct = utils.obs_as_tensor(self.direction[day_no], DEVICE)
            day_data ={
                OBS:obs, TA:ta, BASE:base, PRICE:price, DIRECT:direct}
            days[day_no]=day_data
        self.data = days


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
        obs_rms = self.normalizer[PRICE]
        return (target * np.sqrt (obs_rms.var + self.epsilon)) + obs_rms.mean


    def abs_diff(self, pred, true):
        return np.abs(self.denorm_target(pred)- self.denorm_target(true))

    # def __getitem(self, idx):
    #     if idx >= self.daily_idx[-1]: raise IndexError(idx)
    #
    #     day_no, idx = self.day_search(idx)
    #     day = self.data[day_no]
    #     s_idx = idx
    #     e_idx = idx+self.seq-1
    #     obs = day.data[s_idx: e_idx+1]
    #     ta = day.ta[e_idx -self.ta_seq +1 :e_idx+1]
    #     return obs, ta, day.base, self.direction[day_no][e_idx], self.price[day_no][e_idx]

    def __getitem__(self, idx):
        if idx >= self.daily_idx[-1]: raise IndexError(idx)
        day_no, idx = self.day_search(idx)
        day = self.data[day_no]
        s_idx = idx
        e_idx = idx+self.seq-1
        obs = day[OBS][s_idx: e_idx+1]
        ta = day[TA][e_idx -self.ta_seq +1 :e_idx+1]
        return ({OBS:obs, TA:ta, BASE:day[BASE]}, day[PRICE][e_idx], day[DIRECT][e_idx])

    #

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



