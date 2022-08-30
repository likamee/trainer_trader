import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.positions as positions
import pandas as pd
import talib
from app.utils.indicators import resistance, support
from app.utils.loaders import saver
from oandapyV20.exceptions import V20Error


# realtime
def calcLastSupres(units, close, pair, supres, date, cfg):
    from datetime import datetime, timedelta

    rq = instruments.InstrumentsCandles(instrument=pair[:3]+'_'+pair[3:],
                                        params={"count": cfg['NCANDLESR'][supres-1]+2, "granularity": "d"})
    while True:
        try:
            temp = cfg['API'].request(rq)
        except V20Error as e:
            cfg['LOGGER'].error("v20error: %s", e)
        else:
            datef = datetime.strptime(date[:10], '%y-%m-%d')
            datef = datef - timedelta(days=1)

            temp = pd.DataFrame(temp['candles'])[:-1]
            if datetime.strptime(temp.iloc[-1]['time'][:10], '%y-%m-%d') == datef:
                temp = temp.iloc[:-1]
            else:
                temp = temp.iloc[1:]
            highs = pd.DataFrame(list(temp['mid'].values))['h'].values.astype(float)
            lows = pd.DataFrame(list(temp['mid'].values))['l'].values.astype(float)

            # calc sup
            if units == 1:
                sup = support(lows)
                return True if close < sup else False
            else:
                res = resistance(highs)
                return True if close > res else False


def checktimerange(starttime, endtime, nowtime):
    if starttime < endtime:
        return nowtime >= starttime and nowtime <= endtime
    else:  # over midnight
        return nowtime >= starttime or nowtime <= endtime


def checkmovingpairs(pair, movingpairs, cfg):
    found = 0
    try:
        r = positions.openpositions(accountid=cfg['ACCID'])
        open_positions = cfg['API'].request(r)
    except V20Error as e:
        cfg['LOGGER'].error("v20error: %s", e)
    else:
        if type(open_positions['positions']) == list and len(open_positions['positions']) > 0:
            for position in open_positions['positions']:
                if pair == position['instrument'][:3]+position['instrument'][4:]:
                    found = 1
                    break

        if found == 0:
            movingpairs[pair] = {"atr": 0, "direction": 0, "cume": 0, "tatic": 0}
            saver(movingpairs, 'moving_pairs.txt')

    return movingpairs[pair]


def genmovingpairs(cfg):
    # mount do bd de operacoes
    entries = {}
    for pair in cfg['BARS']:
        entries[pair] = {"atr": 0, "direction": 0, "cume": 0, "tatic": 0}

    try:
        r = positions.OpenPositions(accountID=cfg['ACCID'])
        open_positions = cfg['API'].request(r)
    except V20Error as e:
        cfg['LOGGER'].error("v20error: %s", e)
    else:
        if type(open_positions['positions']) == list and len(open_positions['positions']) > 0:
            for position in open_positions['positions']:
                params_atr = {"count": 100, "granularity": "m30"}
                request = instruments.InstrumentsCandles(instrument=position['instrument'], params=params_atr)
                try:
                    bars = cfg['API'].request(request)
                except V20Error as e:
                    cfg['LOGGER'].error("v20error: %s", e)

                bars = pd.DataFrame(bars['candles'])[:-1]
                closes = pd.DataFrame(list(bars['mid'].values))['c'].values.astype(float)
                highs = pd.DataFrame(list(bars['mid'].values))['h'].values.astype(float)
                lows = pd.DataFrame(list(bars['mid'].values))['l'].values.astype(float)
                atr = talib.ATR(highs, lows, closes, timeperiod=14)
                atr = atr[-1]

                pair = position['instrument'][:3]+position['instrument'][4:]

                if 'tradeids' in position['long']:
                    direction = 1
                else:
                    direction = -1

                entries[pair] = {"atr": atr, "direction": direction, "cume": 0, "tatic": 1}

    saver(entries, 'moving_pairs.txt')
    movingpairs = entries
    return movingpairs


def barscut(bars_temp, k, cfg, train=True):
    a = cfg['NCANDLEST'] if train else cfg['NCANDLESV']
    bars_ret = {}
    for i in bars_temp:
        # dou um limite de uns 200 candles pra processar o safet, nao vou usar tudo, é só pra nao pesar o run
        lim = len(bars_temp[i]['close'].values)
        lim = cfg['NCANDLESSAFET'] if lim - k > cfg['NCANDLESSAFET'] else lim - k
        # bars_ret[i] = bars_temp[i].iloc[k:]
        bars_ret[i] = bars_temp[i].iloc[k:k+cfg['NCANDLESSAFESTART']+a+lim]
        # o corte no teste é: inicio até nteste

    return bars_ret


def frmt(v, instrument):
    if 'jpy' in instrument:
        value = "%.3f" % float(v)
    else:
        value = "%.5f" % float(v)
    return str(value)
