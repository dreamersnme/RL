import os
import sys

import gym
from gym import register

from CONFIG import PRETRAINED
from runner.runner_dict import IterRun
from stable_baselines3 import DDPG, SAC,TD3
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# https://github.com/notadamking/RLTrader/issues/10


names = {"TD3": TD3, "DDPG":DDPG, "SAC":SAC}
def compair_run(iter, model=None):
    noise_set = np.linspace(0.05, 0.6,5)
    nis_start = 5

    if model not in names:targets =[IterRun(TD3, PRETRAINED)]#, IterRun(DDPG), IterRun(SAC)]
    else: targets = [IterRun(names[model])]


    print("======================================")
    print ("======================================")
    print ("======================================")
    for i in range(1, iter):
        noise = None if i <nis_start else noise_set[(i-nis_start) % len(noise_set)]
        for iter_run in targets:
            iter_run.train_eval(noise=noise)



if __name__ == '__main__':
    model = sys.argv[-1].strip()
    print(model)
    compair_run(1000, model.strip())


