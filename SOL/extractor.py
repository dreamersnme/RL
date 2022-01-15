from collections import namedtuple

import pandas as pd
import sqlite3
import numpy as np

from sim.CONFIG import ROOT

pd.set_option('display.max_columns', None)
import talib as tb
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
         ta['MA20'] = tb.MA(c, timeperiod=20) / np.nanmean(tb.MA(c, timeperiod=20))
         ta['vMA5'] = tb.MA(v, timeperiod=5) / np.nanmean(tb.MA(v, timeperiod=5))
         ta['vMA10'] = tb.MA(v, timeperiod=10) / np.nanmean(tb.MA(v, timeperiod=10))
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





def get_base():
    ql = "select * from min_CLK20 where st_dt between '20200305' and '20200403'"
    df_o = pd.read_sql_query (ql, conn)
    df_o['tm_key'] = pd.to_datetime (df_o.tm_key)
    del df_o['dt']
    df_o['h']=df_o['tm_key'].dt.hour
    df_o['h'] = np.where(df_o.h <8, df_o.h+24, df_o.h)
    df_o['m']= df_o['tm_key'].dt.minute

    df_o['mm'] = (df_o.h - 8) * 60 + df_o.m
    df_o['m_diff'] = (df_o['mm'] - df_o['mm'].shift(1)).fillna(0)
    df_o['price'] = df_o['open'] + (df_o['close'] -df_o['open'])*0.8
    df_o['return5'] =  (df_o['price'] - df_o['price'].shift(-5)).fillna(0)




    df = df_o[ (df_o.h >=tm[0]-1) & (df_o.h <=tm[1]+1)]
    base = df_o[ (df_o.h <tm[0])][['st_dt','close','volume']]
    base['prd'] = base['close'] * base['volume']
    base = base.groupby(base.st_dt).sum()
    base["base_price"] = base.prd / base.volume
    base["base_vol"] = base.volume / 1000
    del base["prd"]
    del base["close"]
    del base["volume"]

    print(base.head(50))
    return df, base

def trim_(df):

    return df

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
    del df['open']
    del df['close']
    del df['high']
    del df['low']
    df= pd.merge(left=df, right=base, how="left", on='st_dt')
    df['price'] = df.price - df.base_price
    df['volume'] = df.volume/df.base_vol
    del df['base_price']
    del df['base_vol']
    print ("ADJ HAS NAN", df.isna().sum().values.sum())
    return df


def save():
    df, base = get_data()
    df.to_sql("data_19_28", conn,index=False)
    adj_df = adj(df, base)
    adj_df.to_sql("adj_19_28", conn,index=False)


Day = namedtuple('day', ['data', 'price'])
def load():
    ql = "select * from adj_19_28"
    df = pd.read_sql_query (ql, conn)
    grouped = df.groupby('st_dt')
    days =  [ group.reset_index(drop=True) for _, group in grouped]


    all_days =[]
    for df in days:
        del df['st_dt']
        del df['return5']
        del df['m_diff']
        price = df['price']
        del df['price']
        all_days.append(Day(df.to_numpy(), price.to_numpy()))
    feature_size = len(days[0].columns)

    print(feature_size)
    print(days[0].head(3))
    all_days = all_days[8:15]
    return all_days, len(all_days), feature_size

DATA, DAYS, feature_size = load()

if __name__ == '__main__':
    load()



