# --------------------------- IMPORT LIBRARIES -------------------------
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from gym.utils import seeding
import gym
from gym import spaces
import sim.data_preprocessing as dp
import math


DJI = dp.DJI
DJI_N = dp.DJI_N
CONTEXT_DATA = dp.CONTEXT_DATA
CONTEXT_DATA_N = dp.CONTEXT_DATA_N


NUMBER_OF_STOCKS = len(DJI)

if NUMBER_OF_STOCKS != 1:
    raise Exception ("NEED SINGLE TARGET")

PRICE_FILE = './data/ddpg_WORLD.csv'
INPUT_FILE = './data/ddpg_input_states.csv'
input_states = None
WORLD = None

try:
    input_states = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
    WORLD =  pd.read_csv(PRICE_FILE, index_col='Date', parse_dates=True)
    print("LOAD PRE_PROCESSED DATA")
except :
    print ("LOAD FAIL.  PRE_PROCESSing DATA")
    dataset = dp.DataRetrieval()
    input_states = dataset.get_feature_dataframe(DJI)

    if len(CONTEXT_DATA):
        context_df = dataset.get_feature_dataframe (CONTEXT_DATA)
        input_states = pd.concat([context_df, input_states], axis=1)
    input_states = input_states.dropna()

    input_states.to_csv(INPUT_FILE)
    WORLD = dataset.components_df_o[DJI]
    WORLD.to_csv(PRICE_FILE)

print(input_states.head(3))

# Without context data
#input_states = feature_df

