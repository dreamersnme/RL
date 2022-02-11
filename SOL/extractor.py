from collections import namedtuple

import pandas as pd
import sqlite3

from CONFIG import *

pd.set_option ('display.max_columns', None)
import talib as tb

print (ROOT + "/data/DB/stock_price1.db")
conn = sqlite3.connect (ROOT + "/data/DB/stock_price1.db")
tm = (19, 28)


def ta_idx(df):
    o = df['open'].values
    c = df['close'].values
    h = df['high'].values
    l = df['low'].values
    v = df['volume'].astype (float).values

    # define the technical analysis matrix
    # Most data series are normalized by their series' mean
    ta = pd.DataFrame ()

    for m in [5, 10, 15, 20, 30, 60]:
        value = tb.MA (c, timeperiod=m)
        ta['MA' + str (m)] = value / np.nanmean (value)

    for m in [3, 5, 7, 10, 12]:
        value = tb.MA (v, timeperiod=m)
        ta['vMA' + str (m)] = value / np.nanmean (value)

    for m in [3, 6, 9]:
        value = tb.ADX (h, l, c, timeperiod=m)
        ta['ADX' + str (m)] = value / np.nanmean (value)
        value = tb.ADXR (h, l, c, timeperiod=m)
        ta['ADXR' + str (m)] = value / np.nanmean (value)

    for m in [3, 5, 7, 10, 15]:
        value = tb.ATR (h, l, c, timeperiod=m)
        ta['ATR' + str (m)] = value / np.nanmean (value)


    ta['MACD'] = tb.MACD (c, fastperiod=12, slowperiod=26, signalperiod=9)[0] / \
                 np.nanmean (tb.MACD (c, fastperiod=12, slowperiod=26, signalperiod=9)[0])
    ta['RSI'] = tb.RSI (c, timeperiod=5) / np.nanmean (tb.RSI (c, timeperiod=5))
    ta['BBANDS_U'] = tb.BBANDS (c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0] / \
                     np.nanmean (tb.BBANDS (c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0])
    ta['BBANDS_M'] = tb.BBANDS (c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1] / \
                     np.nanmean (tb.BBANDS (c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1])
    ta['BBANDS_L'] = tb.BBANDS (c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2] / \
                     np.nanmean (tb.BBANDS (c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2])
    ta['AD'] = tb.AD (h, l, c, v) / np.nanmean (tb.AD (h, l, c, v))

    ta['HT_DC'] = tb.HT_DCPERIOD (c) / np.nanmean (tb.HT_DCPERIOD (c))
    ta["High/orice"] = h / o
    ta["Low/Open"] = l / o
    ta["Close/Open"] = c / o
    ta["HL"] = h - l
    return ta


def lag_minmax(close):
    lags = [5, 10, 20, 30, 60]
    columns = []
    for lag in lags:
        columns.extend ([f'minval{lag}', f'minidx{lag}', f'maxval{lag}', f'maxidx{lag}'])

    df = pd.DataFrame (columns=columns, dtype=float)

    for idx in close.index:

        result = []
        for lag in lags:
            target = close.loc[max (0, idx - lag + 1):idx][::-1]
            min_idx = target.idxmin () - idx
            min_val = target.min ()
            max_idx = target.idxmax () - idx
            max_val = target.max ()
            result.extend ([min_val, min_idx, max_val, max_idx])
        df.loc[idx] = result

    return df


def get_dt_base(df_o):
    base = df_o[(df_o.h < tm[0])][['st_dt', 'high', 'low', 'close', 'volume']]
    base['HL'] = base.high - base.low
    del base['high']
    del base['low']

    dt = base.groupby ('st_dt').agg (['mean', 'std', 'min', 'max'])
    dt.columns = dt.columns.map ('_'.join)

    base['prd'] = base['close'] * base['volume']
    base = base.groupby (base.st_dt).sum ()
    # base["base_price"] = base.prd / base.volume
    dt['base_price'] = base.prd / base.volume
    dt['HL'] = dt.close_max - dt.close_min
    dt = dt.reset_index ()

    return dt


def get_base():
    ql = "select * from min_CLK20 where st_dt between '20200205' and '20200403'"
    df_o = pd.read_sql_query (ql, conn)
    df_o['tm_key'] = pd.to_datetime (df_o.tm_key)
    del df_o['dt']
    df_o['h'] = df_o['tm_key'].dt.hour
    df_o['h'] = np.where (df_o.h < 8, df_o.h + 24, df_o.h)
    df_o['m'] = df_o['tm_key'].dt.minute
    df_o['oclock'] = np.where (df_o.m < 30, df_o.m, 60 - df_o.m)
    df_o['oclock'] = 1 / (df_o['oclock'] + 1)

    df_o['mm'] = (df_o.h - 8) * 60 + df_o.m
    df_o['m_diff'] = (df_o['mm'] - df_o['mm'].shift (1)).fillna (0)
    df_o['price'] = df_o['open'] + (df_o['close'] - df_o['open']) * 0.2
    df_o['transaction'] = (df_o['price'].shift (-1)).fillna (0)
    del df_o['price']
    df_o['re1'] = (df_o['transaction'].shift (-1) - df_o['transaction']).fillna (0)
    df_o['re2'] = (df_o['transaction'].shift (-2) - df_o['transaction']).fillna (0)
    df_o['re3'] = (df_o['transaction'].shift (-3) - df_o['transaction']).fillna (0)
    df_o['re4'] = (df_o['transaction'].shift (-4) - df_o['transaction']).fillna (0)
    df_o['re5'] = (df_o['transaction'].shift (-5) - df_o['transaction']).fillna (0)

    df = df_o[(df_o.h >= tm[0] - 2) & (df_o.h <= tm[1] + 1)]
    base = get_dt_base (df_o)
    return df, base


def get_data():
    df, base = get_base ()
    anal = ta_idx (df)
    minmax = lag_minmax (df['close'])
    df = pd.concat ([df.reset_index (drop=True), minmax.reset_index (drop=True), anal.reset_index (drop=True)], axis=1)
    df = df[(df.h >= tm[0]) & (df.h <= tm[1])]
    del df['tm_key']
    del df['h']
    del df['m']

    print ("HAS NAN", df.isna ().sum ().values.sum ())
    return df, base


def adj(df, base):
    df = df.copy ()

    base2 = base[['st_dt', 'base_price', 'volume_mean']]
    df = pd.merge (left=df, right=base2, how="left", on='st_dt')
    df['volume'] = df.volume / df.volume_mean

    prices = ['open', 'high', 'low', 'close', 'transaction', 'maxval5', 'maxval10', 'maxval20', 'minval5', 'minval10', 'minval20']
    for name in prices:
        df[name] = df[name] - df.base_price

    del df['base_price']
    del df['volume_mean']

    print ("ADJ HAS NAN", df.isna ().sum ().values.sum ())
    return df, base


def save():
    df, base = get_data ()

    adj_df, base = adj (df, base)
    df.to_sql ("data_19_28", conn, index=False, if_exists='replace')
    adj_df.to_sql ("adj_19_28", conn, index=False, if_exists='replace')
    base.to_sql ("base_19_28", conn, index=False, if_exists='replace')
    print ("-------------BASE---------------")
    print (base.head (2))
    print ("-------------DATA---------------")
    print (df.head (3))
    print ("-------------ADJ---------------")
    print (adj_df.head (30))


def _load(target, trim):
    ql = "select * from adj_19_28"
    df = pd.read_sql_query (ql, conn)
    grouped = df.groupby ('st_dt')
    days = [(key, group.reset_index (drop=True)) for key, group in grouped]

    all_days = []

    for (st_dt, df) in days:
        price = df[target].to_numpy ().astype (np.float32)
        df = df.drop (columns=trim)
        ql = f"select * from base_19_28 where st_dt = '{st_dt}'"
        base = pd.read_sql_query (ql, conn)
        base = base.drop (columns=['st_dt', 'close_mean', 'close_std', 'close_min', 'close_max'])
        base1 = base.to_numpy ()[0]
        base = np.repeat (base1.reshape (1, -1), repeats=len (df), axis=0)
        obs = np.concatenate ((df, base), axis=-1)

        all_days.append (Day (st_dt, obs.astype (np.float32)
                              , base1.astype (np.float32)
                              , price))
    for i in range (len (all_days) - 1): assert all_days[i].dt < all_days[i + 1].dt
    print (obs.shape)
    print (base1.shape)
    print (price.shape)
    return all_days


def load_ml():
    return _load (['re1', 're2', 're5']
                  , ['st_dt', 'transaction', 'm_diff', 're1', 're2', 're3', 're4', 're5'])


def load_trainset(size=100):
    buff = 4
    test_size = 4
    all_idx = size + test_size + buff

    all_data = load_ml ()[-all_idx: -buff]
    train = all_data[:-test_size]
    test = train[-test_size:]
    tri = all_data[-test_size:]
    valid = test + tri
    print ("TRAIN on: {} DAYS".format (len (train)))
    return train, valid, test, tri


def load_mix(size=100):
    buff = 0
    all_data = load_ml ()[-size:]
    SPEC = DataSpec (all_data)

    print (len (all_data))
    tri_idx = [-1, -4, -7, -10]
    tri_idx.reverse ()
    tri = [all_data[i] for i in tri_idx]
    print (len (tri))
    train = [x for x in all_data if x not in tri]
    check = train[-len (tri_idx):]

    valid = check + tri
    print ("TRAIN on: {} DAYS".format (len (train)))
    print ("TRAIN ON: ", [d.dt for d in train])
    print ("TEST ON: ", [d.dt for d in tri])
    return SPEC, train, valid, check, tri


if __name__ == '__main__':
    save ()
