import os
from collections import OrderedDict, namedtuple
import numpy as np
from gym import spaces
from SOL.normalizer import Standardizer, Normalizer, BothScaler

TRAIN_TARGET =20 #20 #12
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
PREDICTOR = os.path.join(MODEL_DIR, "best_feature_extractor.pt")
Day = namedtuple('day', ['dt', OBS, BASE, PRICE])

# scaler = Normalizer
# scaler = Standardizer
scaler = BothScaler

class DataSpec:
    obs_seq = 17
    ta_seq = 5
    stat_len = 1
    def __init__(self, ori_data):
        ref = ori_data[0]
        self.obs_len = ref.obs.shape[1]
        self.base_len = ref.base.shape[0]
        if scaler == BothScaler:
            self.obs_len = self.obs_len*2
            self.base_len = self.base_len*2

        self.price_len = ref.price.shape[1]
        self.scaler = Scaler(ori_data, self)
        obs = spaces.Box (low=-np.inf, high=np.inf, shape=(self.obs_seq, self.obs_len))
        base = spaces.Box (low=-np.inf, high=np.inf, shape=(self.base_len,))
        stat = spaces.Box (low=-np.inf, high=np.inf, shape=(self.stat_len,))
        self.observation_space = spaces.Dict (OrderedDict ([(OBS, obs), (BASE, base), (STAT, stat)]))
        self.data_space = spaces.Dict (OrderedDict ([(OBS, obs), (BASE, base)]))

    def scaling(self, data): return self.scaler.norm_data(data)
    def denorm_target(self, target): return self.scaler.denormalize(PRICE, target)



class Scaler:
    def __init__(self, data, spec):
        obsN = scaler(shape=(spec.obs_len,))
        baseN = scaler(shape=(spec.base_len,))
        priceN = Standardizer(shape=(spec.price_len,))
        for day in data:
            obsN.update(day.obs)
            baseN.update(day.base)
            priceN.update(day.price)
        self.scalers = {OBS: obsN, BASE: baseN, PRICE: priceN}

    def normalize(self, key, obs):
        return self.scalers[key].norm(obs)

    def denormalize(self, key, data):
        return self.scalers[key].denorm(data)

    def norm_data(self, data):
        data = [Day(ori.dt,  self.scalers[OBS].norm(ori.obs)
                    , self.scalers[BASE].norm(ori.base)
                    , self.scalers[PRICE].norm(ori.price) ) for ori in data]
        return data

    def denorm_data(self, data):
        data = [Day(ori.dt,  self.scalers[OBS].denorm(ori.obs)
                    , self.scalers[BASE].denorm(ori.base)
                    , self.scalers[PRICE].denorm(ori.price) ) for ori in data]
        return data
