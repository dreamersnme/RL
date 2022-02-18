from copy import deepcopy

import torch as th
from torch.utils.data import Dataset
import numpy as np

from CONFIG import *
from SOL import extractor
from stable_baselines3.common import utils
from stable_baselines3.common.running_mean_std import RunningMeanStd


class DLoader(Dataset):
    thresold = 0.09
    def __init__(self, data, spec):
        self.spec = spec
        self.seq = spec.obs_seq
        self.ta_seq = spec.ta_seq
        self.feature_len = spec.obs_len
        self.base_len = spec.base_len
        self.price_len = spec.price_len

        self.day_cnt = len(data)
        self.direction = [self.cal_direction(d.price) for d in data]
        self.daily_size = [day.obs.shape[0] - self.seq+1 for day in data]
        self.daily_idx = np.array([0]+self.daily_size).cumsum()
        self.data = self.up_gpu(spec.scaling(data))

    def cal_direction(self, price):
        plus = np.where(price>=self.thresold, 1, 0)
        mius = np.where(price<=-self.thresold, -1, 0)
        return plus + mius

    def up_gpu(self, data):
        days =dict()
        for day_no, day in enumerate(data):
            obs = utils.obs_as_tensor( day.obs, DEVICE)
            base = utils.obs_as_tensor( day.base, DEVICE)
            price = utils.obs_as_tensor(day.price, DEVICE)
            direct = utils.obs_as_tensor(self.direction[day_no], DEVICE)
            day_data ={
                OBS:obs, BASE:base, PRICE:price, DIRECT:direct}
            days[day_no]=day_data
        return days

    def day_data(self):
        days =[self.data[i] for i in range(len(self.data))]
        return days

    def day_search(self, idx):
        left = 0
        right = self.day_cnt
        while right-left >1:
            mid = int((left + right)/2)
            if idx >= self.daily_idx[mid]: left = mid
            else: right = mid
        idx = idx - self.daily_idx[left]
        return left, idx

    def __getitem__(self, idx):
        if idx >= self.daily_idx[-1]: raise IndexError(idx)
        day_no, idx = self.day_search(idx)
        day = self.data[day_no]
        s_idx = idx
        e_idx = idx+self.seq-1
        obs = day[OBS][s_idx: e_idx+1]
        return ({OBS:obs, BASE:day[BASE]}, day[PRICE][e_idx], day[DIRECT][e_idx])

    def __len__(self):
        return int(self.daily_idx[-1])




class RLDLoader:
    def __init__(self, data, spec):
        self.spec = spec
        self.seq = spec.obs_seq
        self.ta_seq = spec.ta_seq
        self.feature_len = spec.obs_len
        self.base_len = spec.base_len
        self.price_len = spec.price_len

        self.day_cnt = len(data)
        self.daily_size = [day.obs.shape[0] - self.seq+1 for day in data]
        self.daily_idx = np.array([0]+self.daily_size).cumsum()
        data = spec.norm_obs(data)
        self.data = self.make_seq(data)


    @property
    def day_data(self):
        return self.data

    def make_seq(self, data):
        day_data =[]
        for day in data:
            obses =[]
            pricees = []

            for idx in range(day.obs.shape[0] - self.seq+1):
                s_idx = idx
                e_idx = idx + self.seq - 1
                obs = day.obs[s_idx: e_idx + 1]
                obses.append(obs)
                pricees.append(day.price[e_idx])

            day_data.append({OBS: np.array(obses), BASE: day.base,  PRICE:np.array(pricees)})
        return day_data



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



