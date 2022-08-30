#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import math
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import talib
import trendln

# =============================================
# check min, python version
if sys.version_info < (3, 4):
    raise SystemError("QTPyLib requires Python version >= 3.4")

# =============================================
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# =============================================


def highest_rolling(a, n):
    a = pd.Series(a)
    a = a.rolling(int(n)).max()
    return a.values


def lowest_rolling(a, n):
    a = pd.Series(a)
    a = a.rolling(int(n)).min()
    return a.values


def numpy_rolling_window(data, window):
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def numpy_rolling_series(func):
    def func_wrapper(data, window, as_source=False):
        series = data.values if isinstance(data, pd.Series) else data

        new_series = np.empty(len(series)) * np.nan
        calculated = func(series, window)
        new_series[-len(calculated):] = calculated

        if as_source and isinstance(data, pd.Series):
            return pd.Series(index=data.index, data=new_series)

        return new_series

    return func_wrapper


@numpy_rolling_series
def numpy_rolling_mean(data, window, as_source=False):
    return np.mean(numpy_rolling_window(data, window), axis=-1)


@numpy_rolling_series
def numpy_rolling_std(data, window, as_source=False):
    return np.std(numpy_rolling_window(data, window), axis=-1, ddof=1)


@numpy_rolling_series
def numpy_rolling_max(data, window, as_source=False):
    return np.amax(numpy_rolling_window(data, window), axis=-1)


@numpy_rolling_series
def numpy_rolling_min(data, window, as_source=False):
    return np.amin(numpy_rolling_window(data, window), axis=-1)


@numpy_rolling_series
def numpy_rolling_idxmin(data, window, as_source=False):
    return np.argmin(numpy_rolling_window(data, window), axis=-1)+1


@numpy_rolling_series
def numpy_rolling_idxmax(data, window, as_source=False):
    return np.argmax(numpy_rolling_window(data, window), axis=-1)+1


# ---------------------------------------------


def session(df, start='17:00', end='16:00'):
    """ remove previous globex day from df """
    if df.empty:
        return df

    # get start/end/now as decimals
    int_start = list(map(int, start.split(':')))
    int_start = (int_start[0] + int_start[1] - 1 / 100) - 0.0001
    int_end = list(map(int, end.split(':')))
    int_end = int_end[0] + int_end[1] / 100
    int_now = (df[-1:].index.hour[0] + (df[:1].index.minute[0]) / 100)

    # same-dat session?
    is_same_day = int_end > int_start

    # set pointers
    curr = prev = df[-1:].index[0].strftime('%Y-%m-%d')

    # globex/forex session
    if not is_same_day:
        prev = (datetime.strptime(curr, '%Y-%m-%d') -
                timedelta(1)).strftime('%Y-%m-%d')

    # slice
    if int_now >= int_start:
        df = df[df.index >= curr + ' ' + start]
    else:
        df = df[df.index >= prev + ' ' + start]

    return df.copy()

# ---------------------------------------------


def heikinashi(bars):
    bars = bars.copy()
    bars['ha_close'] = (bars['open'] + bars['high'] +
                        bars['low'] + bars['close']) / 4

    # ha open
    bars.loc[:1, 'ha_open'] = (bars['open'] + bars['close']) / 2
    prev_open = bars[:1]['ha_open'].values[0]
    for idx, _ in bars[1:][['ha_open', 'ha_close']].iterrows():
        loc = bars.index.get_loc(idx)
        prev_open = (prev_open + bars['ha_close'].values[loc - 1]) / 2
        bars.loc[loc:loc + 1, 'ha_open'] = prev_open

    bars['ha_high'] = bars.loc[:, ['high', 'ha_open', 'ha_close']].max(axis=1)
    bars['ha_low'] = bars.loc[:, ['low', 'ha_open', 'ha_close']].min(axis=1)

    return pd.DataFrame(index=bars.index,
                        data={'open': bars['ha_open'],
                              'high': bars['ha_high'],
                              'low': bars['ha_low'],
                              'close': bars['ha_close']})

# ---------------------------------------------


def tdi(series, rsi_lookback=13, rsi_smooth_len=2,
        rsi_signal_len=7, bb_lookback=34, bb_std=1.6185):

    rsi_data = rsi(series, rsi_lookback)
    rsi_smooth = sma(rsi_data, rsi_smooth_len)
    rsi_signal = sma(rsi_data, rsi_signal_len)

    bb_series = bollinger_bands(rsi_data, bb_lookback, bb_std)

    return pd.DataFrame(index=series.index, data={
        "rsi": rsi_data,
        "rsi_signal": rsi_signal,
        "rsi_smooth": rsi_smooth,
        "rsi_bb_upper": bb_series['upper'],
        "rsi_bb_lower": bb_series['lower'],
        "rsi_bb_mid": bb_series['mid']
    })

# ---------------------------------------------


def awesome_oscillator(df, weighted=False, fast=5, slow=34):
    midprice = (df['high'] + df['low']) / 2

    if weighted:
        ao = (midprice.ewm(fast).mean() - midprice.ewm(slow).mean()).values
    else:
        ao = numpy_rolling_mean(midprice, fast) - \
            numpy_rolling_mean(midprice, slow)

    return pd.Series(index=df.index, data=ao)


# ---------------------------------------------

def nans(length=1):
    mtx = np.empty(length)
    mtx[:] = np.nan
    return mtx


# ---------------------------------------------

def typical_price(bars):
    res = (bars['high'] + bars['low'] + bars['close']) / 3.
    return pd.Series(index=bars.index, data=res)


# ---------------------------------------------

def mid_price(bars):
    res = (bars['high'] + bars['low']) / 2.
    return pd.Series(index=bars.index, data=res)


# ---------------------------------------------

def ibs(bars):
    """ Internal bar strength """
    res = np.round((bars['close'] - bars['low']) /
                   (bars['high'] - bars['low']), 2)
    return pd.Series(index=bars.index, data=res)


# ---------------------------------------------

def true_range(bars):
    return pd.DataFrame({
        "hl": bars['high'] - bars['low'],
        "hc": abs(bars['high'] - bars['close'].shift(1)),
        "lc": abs(bars['low'] - bars['close'].shift(1))
    }).max(axis=1)


# ---------------------------------------------

def atr(bars, window=14, exp=False):
    tr = true_range(bars)

    if exp:
        res = rolling_weighted_mean(tr, window)
    else:
        res = rolling_mean(tr, window)

    res = pd.Series(res)
    return (res.shift(1) * (window - 1) + res) / window


# ---------------------------------------------

def crossed(series1, series2, direction=None):

    if direction is None or direction == "above":
        above = ((series1 > series2) & (np.roll(series1, 1) < np.roll(series2, 1)))

    if direction is None or direction == "below":
        below = ((series1 < series2) & (np.roll(series1, 1) > np.roll(series2, 1)))

    if direction is None:
        return above or below

    return above if direction == "above" else below


def crossed_above(series1, series2):
    return crossed(series1, series2, "above")


def crossed_below(series1, series2):
    return crossed(series1, series2, "below")

# ---------------------------------------------


def rolling_std(series, window=200, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    if min_periods == window and len(series) > window:
        return numpy_rolling_std(series, window, True)
    else:
        try:
            return series.rolling(window=window, min_periods=min_periods).std()
        except Exception as e:
            return pd.Series(series).rolling(window=window, min_periods=min_periods).std()

# ---------------------------------------------


def rolling_mean(series, window=200, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    if min_periods == window and len(series) > window:
        return numpy_rolling_mean(series, window, True)
    else:
        try:
            return series.rolling(window=window, min_periods=min_periods).mean()
        except Exception as e:
            return pd.Series(series).rolling(window=window, min_periods=min_periods).mean()

# ---------------------------------------------


def rolling_min(series, window=14, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    try:
        return series.rolling(window=window, min_periods=min_periods).min()
    except Exception as e:
        return pd.Series(series).rolling(window=window, min_periods=min_periods).min()


# ---------------------------------------------

def rolling_max(series, window=14, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    try:
        return series.rolling(window=window, min_periods=min_periods).min()
    except Exception as e:
        return pd.Series(series).rolling(window=window, min_periods=min_periods).min()


# ---------------------------------------------

def rolling_weighted_mean(series, window=200, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    try:
        return series.ewm(span=window, min_periods=min_periods).mean()
    except Exception as e:
        return pd.ewma(series, span=window, min_periods=min_periods)


# ---------------------------------------------

def hull_moving_average(series, window=200, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    ma = (2 * rolling_weighted_mean(series, window / 2, min_periods)) - \
        rolling_weighted_mean(series, window, min_periods)
    return rolling_weighted_mean(ma, np.sqrt(window), min_periods)


# ---------------------------------------------

def sma(series, window=200, min_periods=None):
    return rolling_mean(series, window=window, min_periods=min_periods)


# ---------------------------------------------

def wma(series, window=200, min_periods=None):
    return rolling_weighted_mean(series, window=window, min_periods=min_periods)


# ---------------------------------------------

def hma(series, window=200, min_periods=None):
    return hull_moving_average(series, window=window, min_periods=min_periods)


# ---------------------------------------------

def vwap(bars):
    """
    calculate vwap of entire time series
    (input can be pandas series or numpy array)
    bars are usually mid [ (h+l)/2 ] or typical [ (h+l+c)/3 ]
    """
    typical = ((bars['high'] + bars['low'] + bars['close']) / 3).values
    volume = bars['volume'].values

    return pd.Series(index=bars.index,
                     data=np.cumsum(volume * typical) / np.cumsum(volume))


# ---------------------------------------------

def rolling_vwap(bars, window=200, min_periods=None):
    """
    calculate vwap using moving window
    (input can be pandas series or numpy array)
    bars are usually mid [ (h+l)/2 ] or typical [ (h+l+c)/3 ]
    """
    min_periods = window if min_periods is None else min_periods

    typical = ((bars['high'] + bars['low'] + bars['close']) / 3)
    volume = bars['volume']

    left = (volume * typical).rolling(window=window,
                                      min_periods=min_periods).sum()
    right = volume.rolling(window=window, min_periods=min_periods).sum()

    return pd.Series(index=bars.index, data=(left / right)).replace([np.inf, -np.inf], float('NaN')).ffill()


# ---------------------------------------------

def rsi(series, window=14):
    """
    compute the n period relative strength indicator
    """

    # 100-(100/relative_strength)
    deltas = np.diff(series)
    seed = deltas[:window + 1]

    # default values
    ups = seed[seed > 0].sum() / window
    downs = -seed[seed < 0].sum() / window
    rsival = np.zeros_like(series)
    rsival[:window] = 100. - 100. / (1. + ups / downs)

    # period values
    for i in range(window, len(series)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0
        else:
            upval = 0
            downval = -delta

        ups = (ups * (window - 1) + upval) / window
        downs = (downs * (window - 1.) + downval) / window
        rsival[i] = 100. - 100. / (1. + ups / downs)

    # return rsival
    return pd.Series(index=series.index, data=rsival)


# ---------------------------------------------

def macd(series, fast=3, slow=10, smooth=16):
    """
    compute the MACD (Moving Average Convergence/Divergence)
    using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    macd_line = rolling_weighted_mean(series, window=fast) - \
        rolling_weighted_mean(series, window=slow)
    signal = rolling_weighted_mean(macd_line, window=smooth)
    histogram = macd_line - signal
    # return macd_line, signal, histogram
    return pd.DataFrame(index=series.index, data={
        'macd': macd_line.values,
        'signal': signal.values,
        'histogram': histogram.values
    })


# ---------------------------------------------

def bollinger_bands(series, window=20, stds=2):
    ma = rolling_mean(series, window=window, min_periods=1)
    std = rolling_std(series, window=window, min_periods=1)
    upper = ma + std * stds
    lower = ma - std * stds

    return pd.DataFrame(index=series.index, data={
        'upper': upper,
        'mid': ma,
        'lower': lower
    })


# ---------------------------------------------

def weighted_bollinger_bands(series, window=20, stds=2):
    ema = rolling_weighted_mean(series, window=window)
    std = rolling_std(series, window=window)
    upper = ema + std * stds
    lower = ema - std * stds

    return pd.DataFrame(index=series.index, data={
        'upper': upper.values,
        'mid': ema.values,
        'lower': lower.values
    })


# ---------------------------------------------

def returns(series):
    try:
        res = (series / series.shift(1) -
               1).replace([np.inf, -np.inf], float('NaN'))
    except Exception as e:
        res = nans(len(series))

    return pd.Series(index=series.index, data=res)


# ---------------------------------------------

def log_returns(series):
    try:
        res = np.log(series / series.shift(1)
                     ).replace([np.inf, -np.inf], float('NaN'))
    except Exception as e:
        res = nans(len(series))

    return pd.Series(index=series.index, data=res)


# ---------------------------------------------

def implied_volatility(series, window=252):
    try:
        logret = np.log(series / series.shift(1)
                        ).replace([np.inf, -np.inf], float('NaN'))
        res = numpy_rolling_std(logret, window) * np.sqrt(window)
    except Exception as e:
        res = nans(len(series))

    return pd.Series(index=series.index, data=res)


# ---------------------------------------------

def keltner_channel(bars, window=14, atrs=2):
    typical_mean = rolling_mean(typical_price(bars), window)
    atrval = atr(bars, window) * atrs

    upper = typical_mean + atrval
    lower = typical_mean - atrval

    return pd.DataFrame(index=bars.index, data={
        'upper': upper.values,
        'mid': typical_mean.values,
        'lower': lower.values
    })


# ---------------------------------------------

def roc(series, window=14):
    """
    compute rate of change
    """
    res = (series - series.shift(window)) / series.shift(window)
    return pd.Series(index=series.index, data=res)


# ---------------------------------------------

def cci(series, window=14):
    """
    compute commodity channel index
    """
    price = typical_price(series)
    typical_mean = rolling_mean(price, window)
    res = (price - typical_mean) / (.015 * np.std(typical_mean))
    return pd.Series(index=series.index, data=res)


# ---------------------------------------------

def stoch(df, window=14, d=3, k=3, fast=False):
    """
    compute the n period relative strength indicator
    http://excelta.blogspot.co.il/2013/09/stochastic-oscillator-technical.html
    """

    my_df = pd.DataFrame(index=df.index)

    my_df['rolling_max'] = df['high'].rolling(window).max()
    my_df['rolling_min'] = df['low'].rolling(window).min()

    my_df['fast_k'] = 100 * (df['close'] - my_df['rolling_min']) / \
        (my_df['rolling_max'] - my_df['rolling_min'])
    my_df['fast_d'] = my_df['fast_k'].rolling(d).mean()

    if fast:
        return my_df.loc[:, ['fast_k', 'fast_d']]

    my_df['slow_k'] = my_df['fast_k'].rolling(k).mean()
    my_df['slow_d'] = my_df['slow_k'].rolling(d).mean()

    return my_df.loc[:, ['slow_k', 'slow_d']]

# ---------------------------------------------


def zlma(series, window=20, min_periods=None, kind="ema"):
    """
    John Ehlers' Zero lag (exponential) moving average
    https://en.wikipedia.org/wiki/Zero_lag_exponential_moving_average
    """
    min_periods = window if min_periods is None else min_periods

    lag = (window - 1) // 2
    series = 2 * series - series.shift(lag)
    if kind in ['ewm', 'ema']:
        return wma(series, lag, min_periods)
    elif kind == "hma":
        return hma(series, lag, min_periods)
    return sma(series, lag, min_periods)


def zlema(series, window, min_periods=None):
    return zlma(series, window, min_periods, kind="ema")


def zlsma(series, window, min_periods=None):
    return zlma(series, window, min_periods, kind="sma")


def zlhma(series, window, min_periods=None):
    return zlma(series, window, min_periods, kind="hma")

# ---------------------------------------------


def zscore(bars, window=20, stds=1, col='close'):
    """ get zscore of price """
    std = numpy_rolling_std(bars[col], window)
    mean = numpy_rolling_mean(bars[col], window)
    return (bars[col] - mean) / (std * stds)

# ---------------------------------------------


def pvt(bars):
    """ Price Volume Trend """
    trend = ((bars['close'] - bars['close'].shift(1)) /
             bars['close'].shift(1)) * bars['volume']
    return trend.cumsum()

# =================================


def ash(high, low, close, mode, period, smooth, signal, modeMA):
    bulls = bears = smthbulls = smthbears = np.array([0])
    for i in range(1, len(close)):
        # usando preço de close
        price1 = close[i]
        if mode == 0:
            price2 = close[i-1]
            t_bulls = 0.5*(abs(price1-price2)+(price1-price2))
            t_bears = 0.5*(abs(price1-price2)-(price1-price2))
        if mode == 1:
            smax = high[i]
            for j in range(1, period):
                if i-j == 0:
                    break
                smax = max(smax, high[i-j])

            smin = low[i, 1]
            for k in range(1, period):
                if i-j == 0:
                    break
                smin = min(smin, low[i-j])

            t_bulls = price1 - smin
            t_bears = smax - price1

        bulls = np.append(bulls, t_bulls)
        bears = np.append(bears, t_bears)

    # é i ou period?

    if modeMA == 3:
        avgbulls = talib.WMA(bulls, timeperiod=period)
        avgbears = talib.WMA(bears, timeperiod=period)

        # é i ou smooth?
        smthbulls = talib.WMA(avgbulls, timeperiod=smooth)
        smthbears = talib.WMA(avgbears, timeperiod=smooth)

        return smthbulls, smthbears

    elif modaMA == 5:
        # length
        avgbulls = SMMA(bulls, period)
        avgbears = SMMA(bears, period)

        # smooth
        smthbulls = SMMA(avgbulls, smooth)
        smthbears = SMMA(avgbears, smooth)

        # signal
        signbulls = SMMA(smthbulls, signal)
        signbears = SMMA(smthbears, signal)

        return smthbulls, smthbears, signbulls, signbears

    return smthbulls, smthbears


def williamsR(high, low, close, period):
    highest = highest_rolling(high, period)
    lowest = lowest_rolling(low, period)
    wills = (- 100) * (highest - close) / (highest - lowest)

    return wills


def hawkeye(bars, length):

	# [0]open, [1]min, max[2], close[3], close_bid[4],date_time[5]
	rrange=bars['high'].values-bars['low'].values
	rangeAvg=talib.SMA(rrange, timeperiod=length)

	volume = bars['volume'].values
	close = bars['close'].values
	high = bars['high'].values
	low = bars['low'].values
	volumeA=talib.SMA(volume.astype('float64'), timeperiod=length)

	v_color = np.array(['gray'])

	for i in range(1,len(bars['close'].values)):
		high1=high[i-1]
		low1=low[i-1]
		mid1=(high1 + low1)/2

		u1 = mid1 + (high1-low1)/6
		d1 = mid1 - (high1-low1)/6

		r_enabled1 = (rrange[i] > rangeAvg[i] ) and (close[i] < d1) and volume[i] > volumeA[i]
		r_enabled2 = close[i] < mid1
		r_enabled = r_enabled1 or r_enabled2

		g_enabled1 = close[i] > mid1
		g_enabled2 = (rrange[i] > rangeAvg[i]) and (close[i] > u1) and (volume[i] > volumeA[i])
		g_enabled3 = (high[i] > high1) and (rrange[i] < rangeAvg[i]/1.5) and (volume[i] < volumeA[i])
		g_enabled4 = (low[i] < low1) and (rrange[i] < rangeAvg[i]/1.5) and (volume[i] > volumeA[i])
		g_enabled = g_enabled1 or g_enabled2 or g_enabled3 or g_enabled4

		gr_enabled1 = (rrange[i] > rangeAvg[i]) and (close[i] > d1) and (close[i] < u1) and (volume[i] > volumeA[i]) and (volume[i] < volumeA[i]*1.5) and (volume[i] > volume[i-1])
		gr_enabled2 = (rrange[i] < rangeAvg[i]/1.5) and (volume[i] < volumeA[i]/1.5)
		gr_enabled3 = (close[i] > d1) and (close[i] < u1)
		gr_enabled = gr_enabled1 or gr_enabled2 or gr_enabled3

		color= 'gray' if gr_enabled else 'green' if g_enabled else 'red' if r_enabled else 'blue'

		v_color = np.append(v_color, color)
	return v_color, volumeA


def kijun_sen(bars, period_high, period_low):
    result_high = numpy_rolling_max(bars['high'].values, period_high)
    result_low = numpy_rolling_max(bars['low'].values, period_low)
    return (result_high + result_low) / 2


def crossed_multi(series1, series2, direction=None):

    if direction is None or direction == "above":
        above = ((series1 > series2) & (np.roll(series1, 1) > np.roll(series2, 1)) & (np.roll(series1, 2) < np.roll(series2, 2)))

    if direction is None or direction == "below":
        below = pd.Series((series1 < series2) & (np.roll(series1, 1) < np.roll(series2, 1)) & (np.roll(series1, 2) > np.roll(series2, 2)))

    if direction is None:
        return above or below

    return above if direction == "above" else below


#(max[ bb mid[-12] ; bb mid[-2]] - min[ bb mid[-12] ; bb mid[-2]] < 15pips)
#(max[ bb mid[-12] : bb mid[-2]] - min[ bb mid[-12] ; bb mid[-2]] < 15pips)
#esse é pra backtest
def bottleneck_bb(close, mid, period, range_v, pips):
    iter = int(period+range_v)
    bneck = np.array([float(0) for _ in range(iter)])
    pips = (pips/10000) if close[-1] < 10 else (pips/100)
    for i in range(iter, len(close)):
        bneck = np.append(bneck, (1 if (max(mid[i-int(range_v)], mid[i-2]) - min(mid[i-int(range_v)], mid[i-2])) < float(pips) else 0))

    return bneck

#esse só funciona na conta real
def bottleneck_bb_real(close, mid, range_v, pips):
    bneck = np.array([float(0) for _ in range(range_v, len(close))])
    pips = (pips/10000) if close[-1] < 10 else (pips/100)
    for i in range(range_v, len(close)):
        bneck = np.append(bneck, (1 if (max(mid[i-int(range_v)], mid[i-2]) - min(mid[i-int(range_v)], mid[i-2])) < float(pips) else 0))


    return bneck


def pfe(close,n,m):
    n=int(n)
    m=int(m)
    pfepre = np.array([])
    for i in range(n,len(close)):
        soma = np.sqrt((close[i]-close[i-1])**2+1)
        for j in range(1,n-2):
            soma = soma + (np.sqrt((close[i-j]-close[i-j-1])**2+1))
        pfepre= np.append(pfepre,100*np.sqrt((close[i]-close[i-n])**2+n**2)/soma if close[i]-close[i-n]>0 else -100*np.sqrt((close[i]-close[i-n])**2+n**2)/soma)
#terminar: só aplicar media movel -> result -1 e 1

###Proximas duas funcoes sao para o spearman
def rankdata(a):
    temp = a.argsort()
    ranks = np.arange(1,len(a)+1)[temp.argsort()]
    return ranks

def spearman(close, n):
    spear = np.zeros(len(close))
    a = np.zeros(n)
    for i in range(n-1,len(close)): #começa construir em n-1:dps colocar nan
        a = close[i-n+1:i+1]
        rank = rankdata(a)
        d=math.pow((rank[0]-1),2)
        for k in range(2,n+1):
            d+=math.pow((rank[k-1]-k),2)
        spear[i]=1-(6*d/(math.pow(n,3)-n))
    return spear


def ash2(close, period, smooth):
    closediff = np.zeros(len(close))
    bulls = np.zeros(len(close))
    bears = np.zeros(len(close))
    smthbulls = np.zeros(len(close))
    smthbears = np.zeros(len(close))
    closediff=close-np.roll(close,1)
    bulls=talib.WMA(0.5*(abs(closediff)+closediff),period)
    bears=talib.WMA(0.5*(abs(closediff)-closediff),period)
    smthbulls=talib.WMA(bulls,smooth)
    smthbears=talib.WMA(bears,smooth)
    return smthbulls, smthbears


def ashparalelos(BULLS, BEARS):
    REGRESSBULL = (3*BULLS-np.roll(BULLS,1)-np.roll(BULLS,2)-np.roll(BULLS,3))/6
    REGRESSBEAR = (3*BEARS-np.roll(BEARS,1)-np.roll(BEARS,2)-np.roll(BEARS,3))/6
    SIGNAL = np.sign(REGRESSBULL) != np.sign(REGRESSBEAR)
    return SIGNAL

def support(LOW):
    qnt_c = len(LOW)
    minimaIdxs, pmin, mintrend, minwindows = trendln.calc_support_resistance((LOW, None), accuracy=8)
    sup = pmin[0]*qnt_c+pmin[1]
    return sup

def resistance(HIGH):
    qnt_c = len(HIGH)
    maximaIdxs, pmax, maxtrend, maxwindows = trendln.calc_support_resistance((None, HIGH), accuracy=8)
    res = pmax[0]*qnt_c+pmax[1]
    return res


# ---------------------------------------------


def chopiness(bars, window=14):
    atrsum = true_range(bars).rolling(window).sum()
    highs = bars['high'].rolling(window).max()
    lows = bars['low'].rolling(window).min()
    return 100 * np.log10(atrsum / (highs - lows)) / np.log10(window)

# ---------------------------------------------


def mountcandle(bars, jump):
    jump = int(jump)
    bars_j = np.array([{'open': 0, 'high': 0, 'low': 0, 'close': 0}
                       for _ in range(len(bars))])
    for i in range(0, len(bars)-jump, jump):
        temp = {'open': 0, 'high': 0, 'low': 0, 'close': 0}
        temp['open'] = bars['open'].values[i]
        temp['close'] = bars['close'].values[i+jump]
        temp['high'] = np.amax(bars['high'].values[i:i+jump])
        temp['low'] = np.amin(bars['low'].values[i:i+jump])
        bars_j[i:i+jump] = temp
    bars_j = np.roll(bars_j, jump)
    return pd.DataFrame(list(bars_j))


# ----------------------------------------------------

def kuskus(high, low, period, smooth, idxsmth):
    maxs = numpy_rolling_max(high, period)
    mins = numpy_rolling_min(low, period)

    pipsize = (1/10000) if high[-1] < 10 else (1/100)

    #correcao pra quando o lowestlow for maior ou igual ao highest high
    maxs[maxs-mins < (0.1*pipsize)] = mins[maxs-mins < (0.1*pipsize)]+0.1*pipsize
    greatestRange=maxs-mins
    mid=(high+low)/2


    priceLocation=(mid-mins)/greatestRange
    priceLocation= 2.0*priceLocation - 1.0

    smoothed = np.zeros(len(priceLocation))
    fisherIndex = np.zeros(len(priceLocation))
    smoothed[0] = priceLocation[0]

    #pos0 do primeiro buffer é igual a pricelocation
    for i in range(period,len(priceLocation)):
        smoothed[i] = (smooth*smoothed[i-1])+((1.0-smooth)*priceLocation[i])
        smoothed[i] = 0.99 if smoothed[i] > 0.99 else -0.99 if smoothed[i] < -0.99 else smoothed[i]
        if 1-smoothed[i] != 0:
            fisherIndex[i] = math.log((1+smoothed[i])/(1-smoothed[i]))


    smoothedFish = np.zeros(len(fisherIndex))
    for i in range(1,len(fisherIndex)):
        smoothedFish[i] = idxsmth * smoothedFish[i-1]+(1.0-idxsmth)*fisherIndex[i]

    return smoothedFish


# --------------------------------------

def ssl(close,high,low,n):
    smaHigh=talib.MA(high,n)
    smaLow=talib.MA(low,n)
    HLV=np.zeros(len(close))
    ssl_down=np.zeros(len(close))
    ssl_up=np.zeros(len(close))
    for i in range(n-1,len(close)):
        if close[i]>smaHigh[i]:
            HLV[i]=1
        elif close[i]<smaLow[i]:
            HLV[i]=-1
        else:
            HLV[i]=HLV[1]

        if HLV[i]<0:
            ssl_down[i]=smaHigh[i]
            ssl_up[i]=smaLow[i]
        else:
            ssl_down[i]=smaLow[i]
            ssl_up[i]=smaHigh[i]
    return [ssl_down,ssl_up]

# ----------------------------------------

def rex(bars,n,m):
  TVB = 3*bars["close"].values-(bars["high"].values+bars["low"].values+bars["open"].values)
  rex = talib.SMA(TVB,n)
  signal = talib.SMA(rex,m)
  return [rex,signal]


# ----------------------------------------

def ashalma(close,period,dss,m):
    closediff=close-np.roll(close,1)
    bulls=alma(0.5*(abs(closediff)+closediff),period,m,dss)
    bears=alma(0.5*(abs(closediff)-closediff),period,m,dss)
    return bulls,bears
