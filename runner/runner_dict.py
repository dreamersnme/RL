import os
import shutil
import time

from torch.utils.tensorboard import SummaryWriter

from CONFIG import *

from SOL.predictor import Predictor

from runner.callbacks import LearnEndCallback
from sim.env_dy.day_feature_evn import Day_featured
from stable_baselines3.common.noise import NormalActionNoise

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
import numpy as np

ENV = Day_featured

class IterRun:
    MIN_TRADE = 30
    BOOST_SEARCH = 1

    noise_std = 0.4
    adapt_delay = 5
    batch_size = 128
    buffer_size = 1_000_000

    def __init__(self, MODEL, arc=[256,128,64], retrain=False,  seed=None):

        extracted = Predictor()
        self.traindata = extracted.rl_train
        self.valid = extracted.rl_valid
        self.SPEC = extracted.spec
        self.unit_episode = len(self.traindata)
        self.roleout_epi = self.unit_episode * 1

        self.seed = seed
        self.model_cls = MODEL
        self.name = MODEL.__name__
        self.test_env =  ENV(self.valid, self.SPEC, title=self.name, verbose=True, plot_dir="./sFig/{}".format(self.name))
        self.env = self.make_env()
        self.writer = self.tensorboard("./summary_all/{}/".format(self.name))
        self.save = os.path.join(MODEL_DIR, f"ckpt_{self.name}")
        self.buffer = None
        self.arch = arc
        self.iter = 1
        self.time_recoder = TimeRecode(self.writer)
        self.init_grad_epoch = int((self.unit_episode * 600) / self.batch_size)
        if retrain: pass
        elif self.seed is None: self.init_boost (self.MIN_TRADE)
        else :self.set_same()

    def make_env(self):
        env = DummyVecEnv([lambda: ENV(self.traindata, self.SPEC, verbose=False)])
        return VecNormalize(env)

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
        model = self._create(env=env)
        self.train_start = time.time()

        CB = LearnEndCallback ()
        model.learn(total_timesteps=self.roleout_epi, callback=CB, log_interval=self.unit_episode)
        print ("===BOOST FPS: ", CB.fps)
        return model

    def init_boost(self, MIN_TRADE, min_reward=-1000):
        print("-----  BOOST UP", self.name)

        test_env = ENV(self.valid, self.SPEC, verbose=False)
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


    def _create(self, env=None, learning_starts = 1):
        policy_kwargs = dict(net_arch=self.arch)
        noise = NormalActionNoise(
            mean=np.zeros(1), sigma=self.noise_std * np.ones(1)
        )
        if env is None:env = self.env
        seed = self.seed or np.random.randint(1e8)
        model = self.model_cls("MultiInputPolicy", env, verbose=1, action_noise=noise, seed =seed,
                               gradient_steps= self.init_grad_epoch*10, gamma=1.0,
                               batch_size = self.batch_size, policy_kwargs=policy_kwargs, buffer_size=self.buffer_size,
                               learning_starts=learning_starts)




        return model



    def load_model(self, noise):
        model = self.model_cls.load(self.save, env=self.env)
        if self.buffer: model.replay_buffer = self.buffer
        model.set_random_seed (self.seed)
        epoch =  ( 10 + int((model.replay_buffer.size()/self.buffer_size)*20))
        model.gradient_steps = self.init_grad_epoch *epoch

        print("LOADED", self.save, self.iter, model.seed)
        print(self.name, "BUFFER REUSE:", model.replay_buffer.size())
        print(self.name, "TRAIN EPOCH:", epoch)
        if noise is not None:
            model.action_noise.sigma=noise * np.ones(1)
            print(self.name,"Noise Reset:", noise)

        return model

    def fix_weight(self, model):
        for ex in self.extractors(model):
            ex.requires_grad_(False)

    def train_eval(self,  noise=None):
        self.init_env()
        self.time_recoder.start()
        self.seed = np.random.randint (1e8)
        model = self.load_model(noise)

        CB = LearnEndCallback()
        model.learn(total_timesteps=self.roleout_epi, tb_log_name=self.name, callback=CB, log_interval=self.unit_episode)
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
            action, _states = model.predict(obs)
            obs, reward, done, info = env.cont_step(action)
            rewards.append(reward)

        info = info
        rslt = {
            "1_Reward": sum(rewards),
            "2_Profit": info['profit'],
            "3_Risk": info['risk'],
            "4_Trade": info['cnt'],
            "5_Perform": max(0, min(20, info['profit']/info['cnt'])),
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

