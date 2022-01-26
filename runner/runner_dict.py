import os
import shutil
import time

from torch.utils.tensorboard import SummaryWriter
import torch as th
from CONFIG import DataSpec, MODEL_DIR
from SOL import extractor

from runner.callbacks import LearnEndCallback
from sim.env_dy.day_evn import Days
from stable_baselines3.common.noise import NormalActionNoise

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
import numpy as np


data = extractor.load_ml()
valid = data[-10:]
data = data[-15:-5]

ENV = Days
SPEC = DataSpec (data[0])
class IterRun:
    MIN_TRADE = 30
    BOOST_SEARCH = 3
    unit_episode = len(data)
    train_epi = unit_episode * 1
    grad_steps =[(1e5, 2), (5e5, 3), (8e5, 4)]
    noise_std = 0.6
    adapt_delay = 10

    def __init__(self, MODEL, TRANSFER = None, arc=[128, 64], retrain=False, batch_size=128, seed=None):
        self.TRANSFER = None if TRANSFER is None else self.transfer(TRANSFER)

        self.seed = seed
        self.model_cls = MODEL
        self.name = MODEL.__name__
        self.test_env =  ENV(valid, SPEC, title=self.name, verbose=True, plot_dir="./sFig/{}".format(self.name))
        self.env = self.make_env()
        self.writer = self.tensorboard("./summary_all/{}/".format(self.name))
        self.save = os.path.join(MODEL_DIR, f"ckpt_{self.name}")
        self.buffer = None
        self.arch = arc
        self.iter = 1
        self.time_recoder = TimeRecode(self.writer)
        self.batch_size = batch_size
        if retrain: pass
        elif self.seed is None: self.init_boost (self.MIN_TRADE)
        else :self.set_same()

    def transfer(self, trained_file):
        from SOL.model import OutterModel
        pre_trained = OutterModel(SPEC)
        print(trained_file)
        pre_trained.load_state_dict(th.load(trained_file))
        return pre_trained.module

    def make_env(self):
        env = DummyVecEnv([lambda: ENV(data, SPEC, verbose=False)])
        return VecNormalize(env, norm_obs_keys=["obs", "stat"])

    def init_env(self):
        self.test_env.reset_env()
        enves = self.env.venv.envs
        for ee in enves: ee.reset_env()

    def set_same(self):
        model = self.unit_model()
        self.buffer = model.replay_buffer
        model.save(self.save)
        print("-----  CREATE", self.name, " SEED ", self.seed)

    def unit_model(self):
        env = self.make_env()
        model = self._create(env=env, learning_starts=self.unit_episode)
        self.train_start = time.time()
        learn_steps = self.unit_episode * 2

        model.learn(total_timesteps=learn_steps, log_interval=self.unit_episode)
        model.learning_starts = 0
        return model

    def init_boost(self, MIN_TRADE, min_reward=-1000):
        print("-----  BOOST UP", self.name)

        test_env = ENV(valid, SPEC, verbose=False)
        minimum = -1e8
        suit_model = None

        max_cont = -1e8
        bad_model = None

        for iter in range(self.BOOST_SEARCH):
            model = self.unit_model()
            if iter == 0: print (model.policy)
            eval = self.evaluation(model, test_env)
            reward = eval["1_Reward"]
            count = eval['4_Trade']
            if (max_cont < count):
                max_cont = count
                bad_model = model

            if count < MIN_TRADE:
                print (" - - - - - BOOST FAIL: ", self.name, reward, " by Count:", count)
                continue
            print(" - - - - - BOOST PROFIT: ", self.name, reward)
            if (minimum < reward):
                minimum = reward
                suit_model = model
            if reward > min_reward: break
        if suit_model is None:
            suit_model = bad_model
            self.buffer = suit_model.replay_buffer
            print(" - - - - - BOOST Selection Failed: ", self.name, "Bad Model Count", max_cont)

        else:
            print (" - - - - - BOOST Selected: ", self.name, minimum, "Seed:", model.seed)
            self.seed = model.seed
            self.buffer = suit_model.replay_buffer

        suit_model.save(self.save)

        del suit_model


    def _create(self, env=None, learning_starts = 100):
        policy_kwargs = dict(net_arch=self.arch)
        noise = NormalActionNoise(
            mean=np.zeros(1), sigma=self.noise_std * np.ones(1)
        )
        if env is None:env = self.env
        seed = self.seed or np.random.randint(1e8)
        model = self.model_cls("MultiInputPolicy", env, verbose=1, action_noise=noise, seed =seed,
                               gradient_steps= 1, gamma=1.0,
                               batch_size = self.batch_size, policy_kwargs=policy_kwargs, buffer_size=1000_000,
                               learning_starts=learning_starts)


        if self.TRANSFER:
            print(self.name,"TRANSFER LEARNING")
            extractors = self.extractors(model)
            for ex in extractors: ex.load_state_dict(self.TRANSFER.state_dict())
            self.fix_weight(model)

        return model

    def extractors(self, model): return [aa.features_extractor.combined for aa in model.actors]

    def load_model(self, noise):
        model = self.model_cls.load(self.save, env=self.env)
        if self.buffer: model.replay_buffer = self.buffer
        model.set_random_seed (self.seed)
        for grad_on_epi in self.grad_steps:
            if grad_on_epi[0] < model.replay_buffer.size():
                model.gradient_steps = grad_on_epi[1]
            else: break


        print("LOADED", self.save, self.iter, model.seed)
        print(self.name, "BUFFER REUSE:", model.replay_buffer.size())
        if noise is not None:
            model.action_noise.sigma=noise * np.ones(1)
            print(self.name,"Noise Reset:", noise)

        if self.TRANSFER and self.iter <= self.adapt_delay:
            print(self.name," FIX EXTRACTOR :", self.iter ,"<", self.adapt_delay)
            self.fix_weight(model)

        return model

    def fix_weight(self, model):
        for ex in self.extractors(model):
            ex.requires_grad_(False)

    def train_eval(self, traing_epi = None, noise=None):
        self.init_env()
        self.time_recoder.start()
        self.seed = np.random.randint (1e8)
        traing_epi = traing_epi or self.train_epi
        model = self.load_model(noise)
        print(self.name, [list(ee.parameters())[0].requires_grad for ee in self.extractors(model)])

        CB = LearnEndCallback()
        model.learn(total_timesteps=traing_epi, tb_log_name=self.name, callback=CB, log_interval=self.unit_episode)
        self.buffer = model.replay_buffer

        print("===========   EVAL   =======   ", self.name, self.iter, ",FPS: ", CB.fps)
        train = {
            "1_Actor_loss": CB.last_aloss,
            "2_Critic_Loss": CB.last_closs,
            "3_FPS": CB.fps}
        eval = self.evaluation(model, self.test_env)
        self.board("Eval", eval)
        self.board("Train",train)
        self.time_recoder.recode(eval)
        model.save(self.save)
        del model

        self.iter += 1


    def tensorboard(self, dir):
        shutil.rmtree(dir, ignore_errors=True)
        os.makedirs(dir, exist_ok=True)
        return SummaryWriter(dir)

    def board(self, prefix, dict):
        for key, val in dict.items():
            self.writer.add_scalar(prefix + "/" + key, val, self.iter)

    def evaluation(self, model, env = None):
        if not env: env = self.test_env
        obs = env.reset()
        done = False
        rewards = []
        while not done:
            obs = self.env.normalize_obs(obs)
            action, _states = model.predict(obs)
            obs, reward, done, info = env.cont_step(action)
            rewards.append(reward)

        info = info
        rslt = {
            "1_Reward": sum(rewards),
            "2_Profit": info['profit'],
            "3_Risk": info['risk'],
            "4_Trade": info['cnt'],
            "5_Perform": max(0, min(10, info['profit']/info['cnt'])),
        }

        return rslt



class TimeRecode:

    def __init__(self, writer, interval=2):
        self.total_sec=0
        self.tick = 0
        self.writer = writer
        self.interval=interval

    def start(self):
        self.start_tm = time.time()

    def recode(self, dict):
        if dict is None: return False
        self.total_sec += int(time.time()-self.start_tm)
        now = (self.total_sec/60) / self.interval

        if now >= self.tick:
            for key, val in dict.items():
                self.writer.add_scalar("TvIME" + "/" + key, val, now * self.interval)
            self.tick = now + 1
        return True

