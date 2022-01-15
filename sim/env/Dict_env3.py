# --------------------------- IMPORT LIBRARIES -------------------------
from collections import OrderedDict

import numpy as np

from datetime import datetime, timedelta
from gym.utils import seeding
import gym
from gym import spaces
import math

# ------------------------- GLOBAL PARAMETERS -------------------------
# Start and end period of historical data in question
from sim.CONFIG import *
from sim.env import DATA
from sim.epi_plot import EpisodePlot
import time

START_TRAIN = datetime(2008, 12, 31)
END_TRAIN = datetime(2017, 2, 12)
START_TEST = datetime(2017, 2, 12)
END_TEST = datetime(2019, 2, 22)

STARTING_ACC_BALANCE = 0
MAX_TRADE = 2

input_states = DATA.input_states
WORLD = DATA.WORLD

# Without context data
# input_states = feature_df
feature_length = len(input_states.columns)
data_length = len(input_states)

input_np = input_states.to_numpy()


class DictEnv3(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, title="MAIN", verbose=False, plot_dir=None, seq=5):
        self.plot_fig = None if not plot_dir else EpisodePlot(title, plot_dir)
        self.title = title
        self.iteration = 0
        self.step_no = 0
        self.verbose = verbose
        self.seq = seq
        # defined using Gym's Box action space function
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float16)

        # obs = spaces.Box(low = -np.inf, high = np.inf,shape = (2,feature_length),dtype=np.float16)
        obs = spaces.Box(low=-np.inf, high=np.inf, shape=(self.seq, feature_length,))
        stat = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.observation_space = spaces.Dict(OrderedDict([(OBS, obs), (STAT, stat)]))
        # self.reset()

    def reset(self):
        start_time = time.time()
        self.duration = 0
        self.done = False
        self.iteration += 1
        self.reward = 0
        self.buy_price = 0
        self.step_no = 0
        self.day = self.fist_day(START_TRAIN)
        obs = self.get_obs()
        unrealized_pnl = 0.0
        stat = [unrealized_pnl, 0]
        self.state = OrderedDict([(OBS, obs), (STAT, stat)])

        self.up_cnt = 0
        self.down_cnt = 0
        self.total_neg = np.array([], dtype=int)
        self.total_pos = np.array([], dtype=int)
        self.total_commition = 0
        self.unit_log = [0]
        self.acc_balance = [STARTING_ACC_BALANCE]
        self.total_asset = self.acc_balance.copy()
        self.reward_log = [0]
        self.position = 0
        self.position_log = [self.position]
        self.unrealized_asset = [unrealized_pnl]
        self.timeline = [self.day]
        self.duration += time.time() - start_time
        print("   dsfsda ", stat)
        return self.state

    def get_obs(self):
        return input_np[self.step_no - self.seq + 1: self.step_no + 1]

    def fist_day(self, start):
        now = start
        while self.step_no < self.seq - 1:
            now = self.skip_day(now)
            self.step_no = input_states.index.get_loc(now)
        return now

    def skip_day(self, now):
        for add_day in range(1, 100):
            temp_date = now + timedelta(days=add_day)
            if temp_date in input_states.index: return temp_date
            add_day += 1
        raise Exception("NO DAY")

    def get_trade_num(self, normed_action, max_trade):
        action = normed_action + 1
        action = action * (float(2 * max_trade + 1) / 2.0) - max_trade
        action = math.floor(action)
        return min(action, max_trade)

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

        return direction, cleaned, left, clean_all

    def getprice(self, date=None):
        if date is None: date = self.day
        return WORLD.loc[date][0]

    def __trade(self, pre_price, cur_share, new_share, date=None):
        buy_direction, cleaned, left_buy, cleaned_all = self._clean(cur_share, new_share)
        if buy_direction == 0:
            return 0, 0, 0, 0, pre_price

        assert cleaned + left_buy == (new_share - cur_share)
        cost = abs(cleaned + left_buy) * COMMITION

        transacted_price = self.getprice(date) + (SLIPPAGE * buy_direction)  # 살땐 비싸게, 팔땐 싸게

        if cleaned == 0:  # clean은 하지 않았으므로, 같은 방향의 변동
            assert cur_share + left_buy == new_share
            buy_price = (abs(cur_share) * pre_price + abs(left_buy) * transacted_price) / abs(new_share)
            realized = 0

        else:
            realized = -cleaned * (transacted_price - pre_price)
            if not cleaned_all:
                buy_price = pre_price  # self.buy_price[idx] 안바뀜, 일부청산 이기 때문
            else:  # 모두 청산하여 예전 가격 필요없음. 더 거래시 현재 가격
                buy_price = transacted_price
        # print("T ", cleaned, realized, cost, left_buy, buy_price)
        return cleaned, realized, cost, left_buy, buy_price

    def _trade(self, cur_share, action):
        new_share = self.get_trade_num(action, MAX_TRADE)
        cleaned, profit, cost, _, buy_price = self.__trade(self.buy_price, cur_share, new_share)
        cleaned = abs(cleaned)
        thresold = abs(cleaned) * COMMITION
        if profit > thresold:
            self.up_cnt += abs(cleaned)
        elif profit < thresold:
            self.down_cnt += abs(cleaned)
        else:
            pass

        self.buy_price = buy_price
        return new_share, (profit - cost)

    def step_done(self, actions):

        self.step_normal(0)

        total_neg = np.sum(self.total_neg)
        risk_log = -1 * total_neg / np.sum(self.total_pos)
        print("---------------------------", self.iteration - 1)
        if self.verbose:
            print(":::: Iter", self.iteration - 1, self.title
                  , "UP: {}, DOWN: {}, Commition: {}".format(self.up_cnt, self.down_cnt, self.total_commition)
                  , "Acc: {}, Rwd: {}, Neg: {}".format(int(self.total_asset[-1]), int(sum(self.reward_log)),
                                                       int(total_neg)))
            self.render()
        return self.state, self.reward, self.done, {'profit': self.total_asset[-1],
                                                    'risk': risk_log, 'neg': total_neg,
                                                    'cnt': self.down_cnt + self.up_cnt}

    def step(self, actions):
        start_time = time.time()
        self.to_day = self.day
        # assert self.step_no == input_states.index.get_loc (self.to_day)
        self.step_no += 1
        self.done = self.day >= END_TRAIN
        if self.done:
            a, b, c, d = self.step_done(actions[0])
        else:
            a, b, c, d = self.step_normal(actions[0])

        self.duration += time.time() - start_time
        if self.done: print(self.title, "EPI TIME:", self.duration)
        return a, b, c, d

    def _unrealized_profit(self, cur_buy_stat, buy_price, at=None):
        transaction_size = np.sum(abs(cur_buy_stat))
        if transaction_size == 0: return 0
        now = (self.getprice(at) - buy_price) * cur_buy_stat
        now = now - (SLIPPAGE + COMMITION) * transaction_size
        return now

    def step_normal(self, action):
        pre_price = self.buy_price
        balance = self.acc_balance[-1]
        pre_unrealized_pnl = self.state[STAT][0]
        pre_stat = self.state[STAT][-1]

        total_asset_starting = balance + pre_unrealized_pnl

        new_stat, gain = self._trade(pre_stat, action)
        new_bal = balance + gain
        self.acc_balance = np.append(self.acc_balance, new_bal)
        self.position_log = np.append(self.position_log, new_stat)

        # NEXT DAY
        pre_day = self.day
        self.day = self.skip_day(pre_day)
        unrealized_pnl = self._unrealized_profit(new_stat, self.buy_price)
        stat = [unrealized_pnl, new_stat]
        obs = self.get_obs()
        self.state = OrderedDict([(OBS, obs), (STAT, stat)])

        total_asset_ending = new_bal + unrealized_pnl
        step_profit = total_asset_ending - total_asset_starting

        if step_profit < 0:
            self.total_neg = np.append(self.total_neg, step_profit)
        else:
            self.total_pos = np.append(self.total_pos, step_profit)

        self.unit_log = np.append(self.unit_log, step_profit)
        self.unrealized_asset = np.append(self.unrealized_asset, unrealized_pnl)
        self.total_asset = np.append(self.total_asset, total_asset_ending)
        self.timeline = np.append(self.timeline, self.day)

        self.reward = self.cal_reward(new_stat, pre_day, step_profit, pre_unrealized_pnl, pre_price, self.buy_price)
        return self.state, self.reward, self.done, {}

    def cal_reward(self, new_stat, pre_day, step_profit, pre_unrealized_pnl, pre_price, buy_price):
        returns = self.cal_emph_reward(step_profit)
        risk = self.remain_risk(new_stat)
        optimal = self.cal_opt_reward(pre_day, step_profit, pre_unrealized_pnl, pre_price, buy_price)
        reward = returns + 0.01 * risk
        reward += (max(optimal, -5))
        self.reward_log = np.append(self.reward_log, reward)
        return reward

    def remain_risk(self, cur_buy_stat):
        action_power = abs(cur_buy_stat / MAX_TRADE)
        return pow(action_power + 1, 2) - 1

    def get_optimal(self, base_date, base_share, base_unreal, base_price, next_price):
        check_trade = [-MAX_TRADE, 0, MAX_TRADE]
        optimal = self._unrealized_profit(base_share, base_price)

        for target in check_trade:
            if base_share == target: continue
            cleaned, profit, cost, _, buy_price = self.__trade(base_price, base_share, target, base_date)
            profit_sum = profit - cost
            unreal = self._unrealized_profit(target, next_price)
            profit_sum += unreal

            optimal = max(profit_sum, optimal)

        return optimal - base_unreal

    def cal_opt_reward(self, pre_date, profit, pre_unreal, pre_price, next_price):
        opt = self.get_optimal(pre_date, self.position_log[-2], pre_unreal, pre_price, next_price)
        reward = (profit - opt)
        return reward

    def cal_emph_reward(self, profit):
        profit = profit / MAX_TRADE
        if profit < 0:
            profit = min(profit, -1 * pow(abs(profit), 1.5))
        else:
            profit = max(profit, pow(profit, 1.2))
        return profit
        return reward

    def render(self, mode='human'):
        if not self.plot_fig: return self.state
        self.plot_fig.update(iteration=self.iteration - 1, idx=range(len(self.position_log)), pos=self.total_pos,
                             neg=-self.total_neg,
                             cash=self.acc_balance, unreal=self.unrealized_asset, asset=self.total_asset,
                             reward=self.reward_log.cumsum(), position=self.position_log, unit=self.unit_log)
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
