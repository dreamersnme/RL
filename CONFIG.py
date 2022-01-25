import os

TIC = 0.01
TIC_VAL = 1  #10달러 만원
COMMITION = 0.25
SLIPPAGE = 1 #1  # 상방 하방

STAT = "stat"
OBS ="obs"
BASE = 'base'
PRICE = "price"
TA ='ta'
DIRECT = "direct"
DEVICE ="cuda"
INPUT_SET =[STAT, BASE, OBS]

ROOT  = os.path.dirname(os.path.abspath(__file__)+"/../")