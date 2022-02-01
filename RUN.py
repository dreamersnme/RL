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

class TD3X(TD3): pass
class DDPGX(DDPG): pass
class SACX(SAC): pass


names = {"TD3": TD3, "DDPG":DDPG, "SAC":SAC}
namesX = {"TD3X": TD3X, "DDPGX":DDPGX, "SACX":SACX}
def compair_run(iter, model=None):
    noise_set = np.linspace(0.05, 0.3,5)
    nis_start = 5


    if model in names: iter_run = IterRun(names[model])
    elif model in namesX: iter_run = IterRun(namesX[model], PRETRAINED)
    else : iter_run =IterRun(SAC, PRETRAINED)#, IterRun(DDPG), IterRun(SAC)]

    print("======================================")
    print ("======================================")
    print ("======================================")
    for i in range(1, iter):
        noise = None if i <nis_start else noise_set[(i-nis_start) % len(noise_set)]
        iter_run.train_eval(noise=noise)



if __name__ == '__main__':
    model = sys.argv[-1].strip()
    print(model)
    compair_run(1000, model.strip())


