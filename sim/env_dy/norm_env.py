# --------------------------- IMPORT LIBRARIES -------------------------
from collections import OrderedDict

import numpy as np

from gym.utils import seeding
import gym
from gym import spaces
import math

# ------------------------- GLOBAL PARAMETERS -------------------------
# Start and end period of historical data in question
from SOL import extractor
from CONFIG import *
from SOL.extractor import as_gpu_tensor
from SOL.normali import MeanStdNormali
from sim.epi_plot import EpisodePlot
import time
import torch as th

from stable_baselines3.common.running_mean_std import RunningMeanStd

STARTING_ACC_BALANCE = 0
MAX_TRADE = 2


class NormedDays(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data, title="MAIN", verbose=False, plot_dir=None, obs_seq=10, ta_seq =5):
        self.plot_fig = None if not plot_dir else EpisodePlot(title, plot_dir)
        self.title = title
        self.iteration = 0
        self.verbose = verbose
        self.DATA = data
        ref = self.DATA[0]


        self.obs_seq = obs_seq
        self.ta_seq = ta_seq

        feature_len = ref[OBS].shape[1]
        ta_len = ref[TA].shape[1]
        base_len = ref[BASE].shape[0]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float16)
        obs = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_seq, feature_len))
        ta = spaces.Box (low=-np.inf, high=np.inf, shape=(self.ta_seq, ta_len))
        base = spaces.Box(low=-np.inf, high=np.inf, shape=(base_len,))
        stat = spaces.Box (low=-np.inf, high=np.inf, shape=(2,))
        self.observation_data_space = spaces.Dict(OrderedDict([(STAT, stat), (OBS, obs), (TA, ta), (BASE, base)]))

        obs_idx = spaces.Box(low=-np.inf, high=np.inf, shape=(3, ), dtype=np.int)
        ta_idx = spaces.Box (low=-np.inf, high=np.inf, shape=(3, ), dtype=np.int)
        base_idx = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int)
        self.observation_space = spaces.Dict (OrderedDict ([(STAT, stat), (OBS, obs_idx), (TA, ta_idx), (BASE, base_idx)]))

        self.stat_norm = MeanStdNormali(shape=(2,))
        self.reward_norm = MeanStdNormali(shape=(1,))
        self.reset_env()


    def reset_env(self):
        self.day_count = len(self.DATA)
        self.today = 0
        self.acc_balance = [STARTING_ACC_BALANCE]
        self.total_asset = self.acc_balance.copy()
        self.reward_log = [0]
        self.up_cnt = 0
        self.down_cnt = 0
        self.total_neg = np.array([], dtype=float)
        self.total_pos = np.array([], dtype=float)
        self.duration = 0
        self.total_commition = 0
        self.unit_log = [0]
        self.position_log = [0]
        self.unrealized_asset = [0]
        self.price_log =[0]
        self.done = True

    def reset(self):
        if not self.done: raise Exception("Reset without reach end")
        if self.today >= self.day_count:
            self.reset_env()

        self.TAR_DAY = self.today

        TODAY = self.DATA[self.TAR_DAY]
        self.PRICE = (TODAY[PRICE][:,0]/TIC)*TIC_VAL
        self.END = TODAY[OBS].shape[0] - 1 - 1# 마지막인덱스는 최후청산용,신규 obs로 못씀



        self.step_no = self.obs_seq - 1 # 0부터 시작

        unrealized_pnl = 0.0
        self.state = self.get_obs(unrealized_pnl, 0)

        self.done = False
        self.buy_price = 0
        self.today += 1
        self.iteration += 1
        return self.state

    def get_obs(self, unrealized_pnl, pos):
        obs = [self.TAR_DAY, self.step_no - self.obs_seq + 1,  self.step_no + 1]
        ta = [self.TAR_DAY, self.step_no - self.ta_seq + 1, self.step_no + 1]

        self.stat_mem = [unrealized_pnl, pos]
        stat = np.array(self.stat_mem.copy())
        self.stat_norm.update(stat)
        base = [self.TAR_DAY]

        stat_dict = OrderedDict([(STAT, stat), (OBS, obs), (TA, ta), (BASE, base)])
        return stat_dict

    def get_stat_data(self, stat):
        day_no = stat[BASE][0]
        TODAY = self.DATA[day_no]
        base = TODAY[BASE]
        obs = stat[OBS]
        obs = self.DATA[obs[0]][OBS][obs[1]:obs[2]]
        ta = stat[TA]
        ta =  self.DATA[ta[0]][TA][ta[1]:ta[2]]
        stat = self.stat_norm.normalize(stat[STAT])
        stat = as_gpu_tensor(stat)
        stat_dict = OrderedDict([(STAT, stat), (OBS, obs), (TA, ta), (BASE, base)])
        return stat_dict

    def get_stat_data_batch(self, stat):
        obses = stat[OBS]
        obses = [self.DATA[obs[0]][OBS][obs[1]:obs[2]] for obs in obses]
        stat[OBS] = th.stack(obses, dim=0)

        obses = stat[TA]
        obses = [self.DATA[obs[0]][TA][obs[1]:obs[2]] for obs in obses]
        stat[TA] = th.stack(obses, dim=0)

        obses = stat[STAT]
        obses = [as_gpu_tensor(self.stat_norm.normalize(obs)) for obs in obses]
        stat[STAT] = th.stack(obses, dim=0)

        obses = stat[BASE]
        obses = [self.DATA[obs[0]][BASE] for obs in obses]
        stat[BASE] = th.stack(obses, dim=0)

        return stat


    def normalize_reward(self, reword):
        return self.reward_norm.normalize(reword)



    def getprice(self, step=None):
        if step is None: step = self.step_no
        return self.PRICE[step]


    def step_done(self, actions):

        self.step_normal(0)
        total_neg = np.sum(self.total_neg)
        risk_log = -1 * total_neg / np.sum(self.total_pos)
        print("::::", self.today, "/",self.day_count, ", Total days", self.iteration , self.title)

        return self.state, self.reward, self.done, {'profit': self.total_asset[-1],
                                                    'risk': risk_log, 'neg': total_neg,
                                                    'cnt': self.down_cnt + self.up_cnt}

    def cont_step(self, actions):
        obs, reward, done, info =self.step(actions)
        if self.finish_all() :done = True
        elif done:
            obs = self.reset()
            done = False
        else:pass
        return obs, reward, done, info

    def step(self, actions):
        start_time = time.time()
        self.step_no += 1
        self.done = self.step_no == self.END
        if self.done:
            a, b, c, d = self.step_done(actions[0])
        else:
            a, b, c, d = self.step_normal(actions[0])
        self.duration += time.time() - start_time

        if self.finish_all():
            print(self.title, "EPI TIME:", self.duration)
            self.render()

        self.reward_norm.update(np.array([b]))
        return a, b, c, d

    def finish_all(self):
        last_day = True if self.today >= self.day_count else False
        return self.done and last_day


    def _unrealized_profit(self, cur_buy_stat, buy_price):
        if cur_buy_stat == 0: return 0
        now = (self.getprice(self.step_no+1) - buy_price) * cur_buy_stat
        now = now - (SLIPPAGE + COMMITION) * abs(cur_buy_stat)
        return now

    def step_normal(self, action):
        pre_price = self.buy_price
        balance = self.acc_balance[-1]
        pre_unrealized_pnl = self.stat_mem[0]
        pre_stat = self.stat_mem[-1]

        total_asset_starting = balance + pre_unrealized_pnl

        new_stat, gain = self._trade(pre_stat, action)
        new_bal = balance + gain
        self.acc_balance = np.append(self.acc_balance, new_bal)
        self.position_log = np.append(self.position_log, new_stat)



        # NEXT DAY
        unrealized_pnl = self._unrealized_profit(new_stat, self.buy_price)
        stat = [unrealized_pnl, new_stat]


        self.state = self.get_obs(unrealized_pnl, new_stat)
        # test_price = self.getprice(self.step_no - 1)
        # cal = obs[-1][0] + (obs[-1][3] - obs[-1][0]) * 0.2
        # print(test_price - cal)



        total_asset_ending = new_bal + unrealized_pnl
        step_profit = total_asset_ending - total_asset_starting
        if step_profit < 0:
            self.total_neg = np.append(self.total_neg, step_profit)
        else:
            self.total_pos = np.append(self.total_pos, step_profit)

        self.unit_log = np.append(self.unit_log, step_profit)
        self.unrealized_asset = np.append(self.unrealized_asset, unrealized_pnl)
        self.total_asset = np.append(self.total_asset, total_asset_ending)

        self.reward = self.cal_reward(pre_stat, new_stat, step_profit, pre_price)
        return self.state, self.reward, self.done, {}

    def _clean(self, cur_share, new_share):
        shift = new_share - cur_share
        direction = -1 if shift < 0 else 1 if shift > 0 else 0
        clean_all = False

        if cur_share == 0 or shift == 0:
            return direction, 0, shift, clean_all
        if (cur_share < 0 and shift < 0) or (cur_share > 0 and shift > 0):
            return direction, 0, shift, clean_all

        clean_all = (abs(cur_share) <= abs(shift))

        cleaned = -cur_share if clean_all else shift
        left = shift - cleaned
        clean_cnt = abs(cleaned)
        return direction, clean_cnt, left, clean_all

    def __trade(self, pre_price, cur_share, new_share):
        if cur_share == new_share: return 0, 0, pre_price
        buy_direction, clean_cnt, left_buy, cleaned_all = self._clean(cur_share, new_share)
        if buy_direction == 0: return 0, 0, pre_price
        assert (buy_direction*clean_cnt) + left_buy == (new_share - cur_share)

        transacted_price = self.getprice() + (SLIPPAGE * buy_direction)  # 살땐 비싸게, 팔땐 싸게

        if clean_cnt == 0:  # clean은 하지 않았으므로, 같은 방향의 변동
            assert cur_share + left_buy == new_share
            buy_price = (abs(cur_share) * pre_price + abs(left_buy) * transacted_price) / abs(new_share)
            realized = 0
        else:
            commition = (clean_cnt * COMMITION)
            realized = -(buy_direction*clean_cnt) * (transacted_price - pre_price) - commition
            if not cleaned_all: buy_price = pre_price  # self.buy_price[idx] 안바뀜, 일부청산 이기 때문
            else: buy_price = transacted_price # 모두 청산하여 예전 가격 필요없음. 더 거래시 현재 가격

        new_cost = abs(left_buy) * COMMITION
        return realized, new_cost, buy_price

    def get_trade_num(self, normed_action):
        action = normed_action + 1
        action = action * (float(2 * MAX_TRADE + 1) / 2.0) - MAX_TRADE
        action = math.floor(action)
        return min(action, MAX_TRADE)

    def _trade(self, cur_share, action):
        new_share = self.get_trade_num(action)
        realized, new_cost, buy_price = self.__trade(self.buy_price, cur_share, new_share)
        if realized > 0: self.up_cnt += 1
        elif realized < 0: self.down_cnt += 1
        else: pass
        self.buy_price = buy_price
        profit = realized - new_cost
        self.price_log = np.append(self.price_log, self.getprice())
        return new_share, profit

    def cal_reward(self, pre_stat, new_stat, step_profit, pre_price):

        returns = self.cal_emph_reward(step_profit)
        risk = self.remain_risk(new_stat)
        reward = returns #- 0.5 * risk #2틱 risk = 3$ 1틱 1.25
        # optimal = self.cal_opt_reward(step_profit, pre_stat, pre_price, new_stat)

        # reward += (max(optimal, -5))

        self.reward_log = np.append(self.reward_log, reward)
        return reward

    def remain_risk(self, cur_buy_stat):
        action_power = abs(cur_buy_stat / MAX_TRADE)
        return pow(action_power + 1, 2) - 1

    def get_optimal(self, pre_share, pre_price, my_action):
        check_trade = [-MAX_TRADE, 0, MAX_TRADE]
        optimal = -1e6

        for target in check_trade:
            if my_action == target: continue
            realized, new_cost, buy_price = self.__trade(pre_price, pre_share, target)
            unreal = self._unrealized_profit(target, buy_price)
            profit_sum = realized + unreal - new_cost
            optimal = max(profit_sum, optimal)
        return optimal

    def cal_opt_reward(self, my_profit, pre_stat, pre_price, my_action):
        opt = self.get_optimal(pre_stat, pre_price,my_action)
        reward = my_profit - opt
        return reward

    def cal_emph_reward(self, profit):
        profit = profit / MAX_TRADE
        if profit < 0:
            profit = min(profit, -1 * pow(abs(profit), 1.5))
        else:
            profit = max(profit, pow(profit, 1.2))
        return profit


    def render(self, mode='humran'):
        total_neg = 0 if len(self.total_neg) ==0 else np.sum(self.total_neg)

        if self.verbose:
            print( self.title, self.iteration, "UP-: {}, DWN-: {}, Commition: {}".format(self.up_cnt, self.down_cnt, self.total_commition)
                  , "Acc: {}, Rwd: {}, Neg: {}".format(int(self.total_asset[-1]), int(sum(self.reward_log)),
                                                       int(total_neg)))

        adj_price = self.price_log - self.price_log[1]
        adj_reward = self.reward_log.cumsum()

        adj_reward = adj_reward * (  (self.total_asset.max() - self.total_asset.min())/ (adj_reward.max()-adj_reward.min()))
        if not self.plot_fig: return self.state
        self.plot_fig.update(iteration=self.iteration , idx=range(len(self.position_log)), pos=self.total_pos,
                             neg=-self.total_neg,
                             stock=adj_price, unreal=self.unrealized_asset, asset=self.total_asset,
                             reward=adj_reward, position=self.position_log, unit=self.unit_log)
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

