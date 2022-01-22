from collections import namedtuple

import pandas as pd
import sqlite3
import numpy as np

from CONFIG import ROOT

pd.set_option('display.max_columns', None)
import talib as tb
print(ROOT+"/data/DB/stock_price1.db")
conn = sqlite3.connect (ROOT+"/data/DB/stock_price1.db")
tm =(19, 28)

def ta_idx(df):
         o = df['open'].values
         c = df['close'].values
         h = df['high'].values
         l = df['low'].values
         v = df['volume'].astype(float).values

         xx = tb.MA(c, timeperiod=5)

         # define the technical analysis matrix

         # Most data series are normalized by their series' mean
         ta = pd.DataFrame()
         ta['MA5'] = tb.MA(c, timeperiod=5) / np.nanmean(tb.MA(c, timeperiod=5))
         ta['MA10'] = tb.MA(c, timeperiod=10) / np.nanmean(tb.MA(c, timeperiod=10))
         ta['vMA5'] = tb.MA(v, timeperiod=5) / np.nanmean(tb.MA(v, timeperiod=5))
         ta['vMA10'] = tb.MA(v, timeperiod=10) / np.nanmean(tb.MA(v, timeperiod=10))

         ta['MA20'] = tb.MA(c, timeperiod=20) / np.nanmean(tb.MA(c, timeperiod=20))
         ta['vMA20'] = tb.MA(v, timeperiod=20) / np.nanmean(tb.MA(v, timeperiod=20))
         ta['ADX'] = tb.ADX(h, l, c, timeperiod=14) / np.nanmean(tb.ADX(h, l, c, timeperiod=14))
         ta['ADXR'] = tb.ADXR(h, l, c, timeperiod=14) / np.nanmean(tb.ADXR(h, l, c, timeperiod=14))
         ta['MACD'] = tb.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[0] / \
                      np.nanmean(tb.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[0])
         ta['RSI'] = tb.RSI(c, timeperiod=14) / np.nanmean(tb.RSI(c, timeperiod=14))
         ta['BBANDS_U'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0] / \
                          np.nanmean(tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0])
         ta['BBANDS_M'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1] / \
                          np.nanmean(tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1])
         ta['BBANDS_L'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2] / \
                          np.nanmean(tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2])
         ta['AD'] = tb.AD(h, l, c, v) / np.nanmean(tb.AD(h, l, c, v))
         ta['ATR'] = tb.ATR(h, l, c, timeperiod=14) / np.nanmean(tb.ATR(h, l, c, timeperiod=14))
         ta['HT_DC'] = tb.HT_DCPERIOD(c) / np.nanmean(tb.HT_DCPERIOD(c))
         ta["High/orice"] = h / o
         ta["Low/Open"] = l / o
         ta["Close/Open"] = c / o
         ta["HL"] = h-l
         return ta



def get_dt_base(df_o):
    base = df_o[ (df_o.h <tm[0])][['st_dt','high', 'low',  'close','volume']]
    base['HL'] = base.high-base.low
    del base['high']
    del base['low']

    dt = base.groupby('st_dt').agg(['mean', 'std', 'min', 'max'])
    dt.columns = dt.columns.map('_'.join)


    base['prd'] = base['close'] * base['volume']
    base = base.groupby(base.st_dt).sum()
    # base["base_price"] = base.prd / base.volume
    dt['base_price'] = base.prd / base.volume
    dt['HL'] = dt.close_max - dt.close_min
    dt = dt.reset_index()

    return dt


def get_base():
    ql = "select * from min_CLK20 where st_dt between '20200205' and '20200403'"
    df_o = pd.read_sql_query (ql, conn)
    df_o['tm_key'] = pd.to_datetime (df_o.tm_key)
    del df_o['dt']
    df_o['h']=df_o['tm_key'].dt.hour
    df_o['h'] = np.where(df_o.h <8, df_o.h+24, df_o.h)
    df_o['m']= df_o['tm_key'].dt.minute

    df_o['mm'] = (df_o.h - 8) * 60 + df_o.m
    df_o['m_diff'] = (df_o['mm'] - df_o['mm'].shift(1)).fillna(0)
    df_o['price'] = df_o['open'] + (df_o['close'] -df_o['open'])*0.2
    df_o['transaction'] = (df_o['price'].shift(-1)).fillna(0)
    del df_o['price']
    df_o['re1'] = (df_o['transaction'].shift (-1) - df_o['transaction']).fillna (0)
    df_o['re2'] = (df_o['transaction'].shift (-2) - df_o['transaction']).fillna (0)
    df_o['re3'] = (df_o['transaction'].shift (-3) - df_o['transaction']).fillna (0)
    df_o['re4'] = (df_o['transaction'].shift (-4) - df_o['transaction']).fillna (0)
    df_o['re5'] = (df_o['transaction'].shift(-5) -df_o['transaction']).fillna(0)

    df = df_o[ (df_o.h >=tm[0]-1) & (df_o.h <=tm[1]+1)]
    base = get_dt_base(df_o)

    return df, base

def get_data():
    df, base = get_base()
    anal = ta_idx(df)
    df = pd.concat([df.reset_index(drop=True), anal.reset_index(drop=True)], axis=1)
    df = df[(df.h >= tm[0]) & (df.h <= tm[1])]
    del df['tm_key']
    del df['h']
    del df['m']

    print ("HAS NAN", df.isna().sum().values.sum())
    return df, base

def adj(df, base):

    df = df.copy()

    base2 = base[['st_dt','base_price', 'volume_mean']]
    df= pd.merge(left=df, right=base2, how="left", on='st_dt')
    df['volume'] = df.volume / df.volume_mean

    prices = ['open', 'high', 'low', 'close', 'transaction']
    for name in prices:
        df[name] = df[name] - df.base_price

    del df['base_price']
    del df['volume_mean']

    print ("ADJ HAS NAN", df.isna().sum().values.sum())
    return df, base


def save():
    df, base = get_data()

    adj_df, base = adj(df, base)
    df.to_sql("data_19_28", conn, index=False, if_exists='replace')
    adj_df.to_sql("adj_19_28", conn,index=False,if_exists='replace')
    base.to_sql("base_19_28", conn, index=False, if_exists='replace')
    print("-------------BASE---------------")
    print(base.head(2))
    print("-------------DATA---------------")
    print(df.head(3))
    print("-------------ADJ---------------")
    print(adj_df.head(3))

Day = namedtuple('day', ['dt', 'data', 'ta', 'price', 'base'])
def load_ori():
    ql = "select * from data_19_28"
    df = pd.read_sql_query (ql, conn)
    grouped = df.groupby('st_dt')
    days =  [ group.reset_index(drop=True) for _, group in grouped]


    all_days =[]
    for df in days:
        del df['st_dt']
        del df['return5']
        del df['m_diff']
        price = df['transaction']
        del df['transaction']
        all_days.append(Day(df.to_numpy(), price.to_numpy()))
    feature_size = len(days[0].columns)

    print(feature_size)
    print(days[0].head(3))
    return all_days, feature_size

def _load(target, trim ):
    ql = "select * from adj_19_28"
    df = pd.read_sql_query (ql, conn)
    grouped = df.groupby('st_dt')
    days =[ (key, group.reset_index(drop=True)) for key, group in grouped]

    all_days =[]

    for (st_dt, df) in days:
        price = df[target]
        df =df.drop (columns=trim)
        ql = f"select * from base_19_28 where st_dt = '{st_dt}'"
        base = pd.read_sql_query(ql, conn)
        base = base.drop(columns=[ 'st_dt', 'close_mean',  'close_std',  'close_min',  'close_max'])
        obs_cols = [ 'open','high' , 'low','mm', 'close', 'volume', 'HL']
        ta_cols = [i for i in df.columns if i not in obs_cols]

        obs = df[obs_cols]
        ta = df[ta_cols]

        all_days.append(Day(st_dt, obs.to_numpy().astype(np.float32)
                            , ta.to_numpy().astype(np.float32)
                            , price.to_numpy().astype(np.float32)
                            , base.to_numpy().astype(np.float32)[0]))
    for i in range(len(all_days)-1): assert all_days[i].dt < all_days[i+1].dt
    feature_size = len(df.columns)
    base_size = len(base.columns)

    print(feature_size, base_size)
    print("-----OBS")
    print(obs.head(2))
    print("-----TA")
    print(ta.head(2))
    print("-----BASE")
    print(base.head())
    print("-----PRICE")
    print(price.head(2))
    return all_days, feature_size, base_size

def load_ml():
    all_days, feature_size, base_size = _load(['re1', 're2', 're3', 're4', 're5']
                 , ['st_dt','transaction','m_diff' ,'re1', 're2', 're3', 're4', 're5' ])
    TRAIN, _, TEST = split(all_days)
    return TRAIN, TEST

def load():
    all_days, feature_size,  all_days, feature_size, base_size = _load('transaction'
                 , ['st_dt', 'transaction','return5','m_diff' , 'open','high' , 'low'])
    TRAIN, _, TEST = split(all_days)
    return TRAIN, TEST

def split(all_days):
    idxes = list(range(len(all_days)))
    # e_day = idxes[-5:]
    # extra = [3,10,14,17]
    # e_day = list(set(extra+e_day))
    # t_day = [i for i in idxes if i not in e_day]
    t_day = idxes[:-5]
    e_day = idxes[-8:]
    print("TAIN ON:",t_day)
    print("TEST ON:", e_day)

    train = [all_days[i] for i in t_day]
    test = [all_days[i] for i in e_day]
    return train, len(train), test


if __name__ == '__main__':
    save()






