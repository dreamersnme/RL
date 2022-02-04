import os
from collections import OrderedDict

import numpy as np
from gym import spaces


TRAIN_TARGET = 5
TIC = 0.01
TIC_VAL = 1  #10달러 만원
COMMITION = 0.25
SLIPPAGE = 1 #1  # 상방 하방
STARTING_ACC_BALANCE = 0
MAX_TRADE = 2

STAT = "stat"
OBS ="obs"
BASE = 'base'
PRICE = "price"
DIRECT = "direct"
DEVICE ="cuda"
INPUT_SET =[STAT, BASE, OBS]

ROOT  = os.path.dirname(os.path.abspath(__file__)+"/../")
MODEL_DIR = os.path.join(ROOT, "MODELS")
PRETRAINED = os.path.join(MODEL_DIR, "best_model.pt")

class DataSpec:
    obs_seq = 21
    ta_seq = 5
    stat_len = 1
    def __init__(self, ref):
        self.obs_len = ref.obs.shape[1]
        self.base_len = ref.base.shape[0]
        self.price_len = ref.price.shape[1]

        obs = spaces.Box (low=-np.inf, high=np.inf, shape=(self.obs_seq, self.obs_len))
        base = spaces.Box (low=-np.inf, high=np.inf, shape=(self.base_len,))
        stat = spaces.Box (low=-np.inf, high=np.inf, shape=(self.stat_len,))
        self.observation_space = spaces.Dict (OrderedDict ([(OBS, obs), (BASE, base), (STAT, stat)]))
        self.data_space = spaces.Dict (OrderedDict ([(OBS, obs), (BASE, base)]))
