import torch as th
from torch.utils.data import Dataset
import numpy as np

from CONFIG import *
from SOL import extractor


class DLoader(Dataset):
    def __init__(self, data, seq = 2):
        self.data = data
        self.day_cnt = len(data)
        # self.obs = [d.data for d in data]
        # self.base = [d.base for d in data]
        # self.target = [d.price for d in data]
        self.seq = seq
        self.daily_size = [day.data.shape[0] - seq+1 for day in self.data]
        self.daily_idx = np.array([0]+self.daily_size).cumsum()

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
        day, idx = self.day_search(idx)
        day = self.data[day]
        s_idx = idx
        e_idx = idx+self.seq-1
        obs = day.data[s_idx: e_idx+1]
        return {"Y": day.price[e_idx], "X":{OBS:obs,BASE: day.base}}

    def __len__(self):
        return self.daily_idx[-1]


if __name__ == "__main__":
    loader = DLoader(extractor.load_ml())
    print(loader.__len__())

    print (loader.__getitem__ (11352))

    # print(loader.day_search(0))
    # print (loader.day_search (593))
    # print (loader.day_search (594))
    # print (loader.day_search (1188))
    # print (loader.day_search (8900))
    # print (loader.day_search (8901))



