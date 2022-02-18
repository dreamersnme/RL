from CONFIG import *
from SOL import extractor
from SOL.DLoader import DLoader, RLDLoader
from SOL.extractor import Extractor
from SOL.sol_train import CECK
import torch as th

class Predictor:
    def __init__(self, model=None):
        REF, t_data, valid, test, tri = Extractor.load()
        t_data = t_data + tri[:3]

        self.spec = REF
        self.price_model = CECK().load(REF) if model is None else model
        self.feature_model = self.price_model.feature_ext
        self.train = RLDLoader(t_data, REF).day_data
        self.valid = RLDLoader(valid, REF).day_data
        self.rl_train = self.convert(self.train)
        self.rl_valid = self.convert(self.valid)
        #
        #
        # for date in self.train:
        #     date[OBS] = self.feature_model.predict(REF.norm_obs(date))
        #
        # for date in self.valid:
        #     date[OBS] = self.feature_model.predict(REF.norm_obs(date))


    def convert(self, data):
        return [Day(None, self.feature_model.predict(ori)
                , ori[BASE]
                , ori[PRICE]) for ori in data]

    def price_test(self):
        all_cnt = 0
        sum = 0

        for day in self.valid:
            pred = self.feature_model.predict(day)
            pred = self.price_model.predict_encoded(pred)
            pred = self.spec.denorm_target(pred)
            true = day[PRICE]
            diff = np.sum(np.abs(pred[:, 0] - true[:, 0]))
            sum += np.sum(diff)
            all_cnt+= true.shape[0]

        print('PRED diff: {:>.3}'.format(sum/all_cnt))


    #
    # def price_test(self):
    #     all_cnt = 0
    #     sum = 0
    #     with th.no_grad():
    #         for day in self.valid:
    #             # pred = self.feature_model.predict(day)
    #             # pred = self.price_model.predict_encoded(pred)
    #             pred = self.price_model.predict(day)
    #             pred = pred.cpu().numpy()
    #             pred = self.spec.denorm_target(pred)
    #             true = day[PRICE]
    #             diff = np.sum(np.abs(pred[:, 0] - true[:, 0]))
    #             sum += np.sum(diff)
    #             all_cnt+= true.shape[0]
    #
    #     print('PRED diff: {:>.3}'.format(sum/all_cnt))