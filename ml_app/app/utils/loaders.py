import csv
import pickle

import app.strategy.meuchapa as stg
import numpy as np
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
import talib
from app.utils.indicators import resistance, support
from oandapyV20.contrib.factories import InstrumentsCandlesFactory


def downloadCandles(cfg):
    dataupdate = False
    params = {
        "from": "2020-01-01T00:00:00Z",
        "to": "2022-08-31T00:00:00Z",
        "granularity": 'M30',
        "price": 'BAM'
    }

    for i in cfg['BARS']:
        print('downloading '+i+' data')
        if dataupdate:
            cfg['BARS'][i] = pd.read_csv(
                "data/wrangling/30m/"+i+".csv", index_col=0, usecols=[0, 1, 2, 3, 4])
            # vou pegar 15 posicoes atras pra calcular o atr!
            data = cfg['BARS'][i].tail(15).index.values[0]
            params['from'] = data.replace(' ', 't')+'z'
            cfg['BARS'][i] = 0

        for r in InstrumentsCandlesFactory(instrument=i[:round(len(i)/2)]+'_'+i[round(len(i)/2):], params=params):
            oanda_bars = cfg['API'].request(r)
            if len(oanda_bars['candles']) <= 0:
                continue

            oanda_time = [x['time'] for x in oanda_bars['candles']]
            oanda_data_bid = [x['bid'] for x in oanda_bars['candles']]
            oanda_data_ask = [x['ask'] for x in oanda_bars['candles']]
            oanda_data_mid = [x['mid'] for x in oanda_bars['candles']]
            for idx, x in enumerate(oanda_bars['candles']):
                oanda_data_mid[idx].update({'volume': x['volume']})

            bars_temp_bid = pd.DataFrame(oanda_data_bid, index=oanda_time)
            bars_temp_ask = pd.DataFrame(oanda_data_ask, index=oanda_time)
            bars_temp_mid = pd.DataFrame(oanda_data_mid, index=oanda_time)

            bars_temp_bid = bars_temp_bid.rename(
                columns={'c': 'close_bid', 'h': 'high_bid', 'l': 'low_bid', 'o': 'open_bid'})
            bars_temp_ask = bars_temp_ask.rename(
                columns={'c': 'close_ask', 'h': 'high_ask', 'l': 'low_ask', 'o': 'open_ask'})
            bars_temp_mid = bars_temp_mid.rename(
                columns={'c': 'close', 'h': 'high', 'l': 'low', 'o': 'open', 'volume': 'volume'})

            bars_temp = pd.concat(
                [bars_temp_bid, bars_temp_ask, bars_temp_mid], axis=1, ignore_index=False)

            bars_temp[['close_bid', 'high_bid', 'low_bid', 'open_bid', 'close_ask', 'high_ask', 'low_ask', 'open_ask',
                       'close', 'high', 'low', 'open', 'volume']] = \
                bars_temp[['close_bid', 'high_bid', 'low_bid', 'open_bid', 'close_ask', 'high_ask', 'low_ask',
                           'open_ask', 'close', 'high', 'low', 'open', 'volume']].apply(pd.to_numeric)

            if isinstance(cfg['BARS'][i], int):
                cfg['BARS'][i] = bars_temp
            else:
                cfg['BARS'][i] = cfg['BARS'][i].append(bars_temp)

            cfg['BARS'][i]['atr'] = talib.ATR(cfg['BARS'][i]['high'].values, cfg['BARS'][i]['low'].values,
                                              cfg['BARS'][i]['close'].values, timeperiod=14)

        # bars[i] = bars[i][r+1:]
        cfg['BARS'][i].index = cfg['BARS'][i].index.str.replace('T', ' ').str.replace('.000000000Z', '')

        if not dataupdate:
            with open('data/wrangling/30m/' + i + '.csv', 'w') as filedata:
                writer = csv.writer(filedata, delimiter=',')
                header = ['data', 'open_ask', 'close_ask', 'high_ask', 'low_ask', 'open_bid',
                          'close_bid', 'high_bid', 'low_bid', 'open', 'close', 'high', 'low', 'volume', 'atr']
                # header.extend(list(cor.keys()))
                writer.writerow(header)
        else:
            cfg['BARS'][i] = cfg['BARS'][i].iloc[1:]

        for k in range(len(cfg['BARS'][i]['close'].values)):
            with open('data/wrangling/30m/' + i + '.csv', 'a') as filedata:
                writer = csv.writer(filedata, delimiter=',')

                # cor = [cfg['BARS'][i][j].values[k] for j in bars]
                data = [str(cfg['BARS'][i].index.values[k]), str(cfg['BARS'][i]['open_ask'].values[k]),
                        str(cfg['BARS'][i]['close_ask'].values[k]), str(cfg['BARS'][i]['high_ask'].values[k]),
                        str(cfg['BARS'][i]['low_ask'].values[k]), str(cfg['BARS'][i]['open_bid'].values[k]),
                        str(cfg['BARS'][i]['close_bid'].values[k]), str(cfg['BARS'][i]['high_bid'].values[k]),
                        str(cfg['BARS'][i]['low_bid'].values[k]), str(cfg['BARS'][i]['open'].values[k]),
                        str(cfg['BARS'][i]['close'].values[k]), str(cfg['BARS'][i]['high'].values[k]),
                        str(cfg['BARS'][i]['low'].values[k]), str(cfg['BARS'][i]['volume'].values[k]),
                        str(cfg['BARS'][i]['atr'].values[k])]
                # data.extend(cor)
                # data = [date, str(cfg['BARS'][i]['close_ask'][k]), str(cfg['BARS'][i]['high_ask'][k]),
                #         str(cfg['BARS'][i]['low_ask'][k])]
                writer.writerow(data)


def downloadSupres(cfg):
    aux_params = {
        "from": "2020-01-01T00:00:00Z",
        "to": "2022-08-31T00:00:00Z",
        "granularity": 'D',
        "price": 'M'
    }

    # supres data
    for j in cfg['BARS']:
        cfg['BARS'][j] = 0
        for r in InstrumentsCandlesFactory(instrument=j[:round(len(j)/2)]+'_'+j[round(len(j)/2):], params=aux_params):
            try:
                oanda_bars = cfg['API'].request(r)
            except Exception as e:
                print(e)
            if len(oanda_bars['candles']) <= 0:
                continue

            oanda_time = [x['time'] for x in oanda_bars['candles']]
            oanda_data_mid = [x['mid'] for x in oanda_bars['candles']]
            """ for idx, x in enumerate(oanda_bars['candles']):
                oanda_data_mid[idx].update({'volume': x['volume']}) """

            bars_temp_mid = pd.DataFrame(oanda_data_mid, index=oanda_time)
            bars_temp_mid = bars_temp_mid.rename(
                columns={'c': 'close', 'h': 'high', 'l': 'low', 'o': 'open'})

            bars_temp = bars_temp_mid

            bars_temp[['close', 'high', 'low', 'open']] = bars_temp[[
                'close', 'high', 'low', 'open']].apply(pd.to_numeric)

            if isinstance(cfg['BARS'][j], int):
                cfg['BARS'][j] = bars_temp
            else:
                cfg['BARS'][j] = cfg['BARS'][j].append(bars_temp)

        cfg['BARS'][j].index = cfg['BARS'][j].index.str.replace('T', ' ').str.replace('.000000000Z', '').str[0:10]
        cfg['BARS'][j] = cfg['BARS'][j][~cfg['BARS'][j].index.duplicated()]

    sup, res = np.zeros(5, dtype=object), np.zeros(5, dtype=object)

    for i in cfg['BARS']:
        with open('data/wrangling/supresd/' + i + '.csv', 'w') as filedata:
            writer = csv.writer(filedata, delimiter=',')
            header = ['data', 'support30', 'resistance30', 'support50', 'resistance50', 'support100', 'resistance100',
                      'support150', 'resistance150', 'support200', 'resistance200']
            # header.extend(list(cor.keys()))
            writer.writerow(header)

        for j in range(max(cfg['NCANDLESR']), len(cfg['BARS'][i])):
            for k in range(len(cfg['NCANDLESR'])):
                sup[k] = support(cfg['BARS'][i].iloc[j-cfg['NCANDLESR'][k]:j]['low'].values)
                res[k] = resistance(cfg['BARS'][i].iloc[j-cfg['NCANDLESR'][k]:j+1]['high'].values)

            with open('data/wrangling/supresd/' + i + '.csv', 'a') as filedata:
                writer = csv.writer(filedata, delimiter=',')

                data = [str(cfg['BARS'][i].index.values[j]), str(sup[0]), str(res[0]), str(sup[1]),
                        str(res[1]), str(sup[2]), str(res[2]), str(sup[3]), str(res[3]), str(sup[4]), str(res[4])]
                # data = [date, str(cfg['BARS'][i]['close_ask'][k]), str(cfg['BARS'][i]['high_ask'][k]),
                #        str(cfg['BARS'][i]['low_ask'][k])]
                writer.writerow(data)


def loadBars(bars, cfg, params=False):
    if cfg['MODE'] == 'training' or cfg['MODE'] == 'realtime':
        for i in bars:
            # bars normal
            bars[i] = 0
            while True:
                r = instruments.InstrumentsCandles(instrument=i[:round(len(i)/2)]+'_'+i[round(len(i)/2):],
                                                   params=params)
                try:
                    oanda_bars = cfg['API'].request(r)
                    break
                except Exception as e:
                    print(e)
                    continue
            if len(oanda_bars['candles']) <= 0:
                continue

            oanda_time = [x['time'] for x in oanda_bars['candles']]
            oanda_data_bid = [x['bid'] for x in oanda_bars['candles']]
            oanda_data_ask = [x['ask'] for x in oanda_bars['candles']]
            oanda_data_mid = [x['mid'] for x in oanda_bars['candles']]
            for idx, x in enumerate(oanda_bars['candles']):
                oanda_data_mid[idx].update({'volume': x['volume']})

            bars_temp_bid = pd.DataFrame(oanda_data_bid, index=oanda_time)
            bars_temp_ask = pd.DataFrame(oanda_data_ask, index=oanda_time)
            bars_temp_mid = pd.DataFrame(oanda_data_mid, index=oanda_time)

            bars_temp_bid = bars_temp_bid.rename(
                columns={'c': 'close_bid', 'h': 'high_bid', 'l': 'low_bid', 'o': 'open_bid'})
            bars_temp_ask = bars_temp_ask.rename(
                columns={'c': 'close_ask', 'h': 'high_ask', 'l': 'low_ask', 'o': 'open_ask'})
            bars_temp_mid = bars_temp_mid.rename(
                columns={'c': 'close', 'h': 'high', 'l': 'low', 'o': 'open', 'volume': 'volume'})

            bars_temp = pd.concat(
                [bars_temp_bid, bars_temp_ask, bars_temp_mid], axis=1, ignore_index=False)

            bars_temp[['close_bid', 'high_bid', 'low_bid', 'open_bid', 'close_ask', 'high_ask', 'low_ask', 'open_ask',
                       'close', 'high', 'low', 'open', 'volume']] = \
                bars_temp[['close_bid', 'high_bid', 'low_bid', 'open_bid', 'close_ask', 'high_ask', 'low_ask',
                           'open_ask', 'close', 'high', 'low', 'open', 'volume']].apply(pd.to_numeric)

            if isinstance(bars[i], int):
                bars[i] = bars_temp
            else:
                bars[i] = bars[i].append(bars_temp)

            bars[i]['atr'] = talib.ATR(bars[i]['high'].values, bars[i]['low'].values,
                                       bars[i]['close'].values, timeperiod=14)

            bars[i] = bars[i].iloc[14:]
            bars[i].index = bars[i].index.str.replace('T', ' ').str.replace('.000000000Z', '')
            bars[i] = bars[i][~bars[i].index.duplicated()]

    elif cfg['MODE'] == 'backtest':
        for i in bars:
            print('loading '+i+' data')
            cols = list(range(15))
            bars[i] = pd.read_csv("data/wrangling/30m/"+i+".csv", index_col=0, usecols=cols)
            bars[i] = bars[i][~bars[i].index.duplicated()]
    return bars


def loadSupres(aux_bars, cfg, aux_params=False):
    # supres data
    if cfg['MODE'] == 'training' or cfg['MODE'] == 'realtime':
        for j in cfg['BARS']:
            aux_bars[j] = 0
            for r in InstrumentsCandlesFactory(instrument=j[:round(len(j)/2)]+'_'+j[round(len(j)/2):],
                                               params=aux_params):
                try:
                    oanda_bars = cfg['API'].request(r)
                except Exception as e:
                    print(e)
                if len(oanda_bars['candles']) <= 0:
                    continue

                oanda_time = [x['time'] for x in oanda_bars['candles']]
                oanda_data_mid = [x['mid'] for x in oanda_bars['candles']]

                bars_temp_mid = pd.DataFrame(oanda_data_mid, index=oanda_time)
                bars_temp_mid = bars_temp_mid.rename(
                    columns={'c': 'close', 'h': 'high', 'l': 'low', 'o': 'open'})

                bars_temp = bars_temp_mid

                bars_temp[['close', 'high', 'low', 'open']] = bars_temp[[
                    'close', 'high', 'low', 'open']].apply(pd.to_numeric)

                if isinstance(aux_bars[j], int):
                    aux_bars[j] = bars_temp
                else:
                    aux_bars[j] = aux_bars[j].append(bars_temp)

            aux_bars[j].index = aux_bars[j].index.str.replace('T', ' ').str.replace('.000000000Z', '').str[0:10]

            aux_bars[j] = aux_bars[j][~aux_bars[j].index.duplicated()]

        supres = pd.DataFrame(columns=['data', 'support30', 'resistance30', 'support50', 'resistance50', 'support100',
                                       'resistance100', 'support150', 'resistance150', 'support200', 'resistance200'])
        sup, res = np.zeros(5, dtype=object), np.zeros(5, dtype=object)
        for i in aux_bars:
            print('calculating '+i+' supres')
            for j in range(max(cfg['NCANDLESR']), len(aux_bars[i])):
                for k in range(len(cfg['NCANDLESR'])):
                    sup[k] = support(aux_bars[i].iloc[j-cfg['NCANDLESR'][k]:j]['low'].values)
                    res[k] = resistance(aux_bars[i].iloc[j-cfg['NCANDLESR'][k]:j+1]['high'].values)

                supres.loc[j] = [str(aux_bars[i].index.values[j]), sup[0], res[0], sup[1],
                                 res[1], sup[2], res[2], sup[3], res[3], sup[4], res[4]]

            aux_bars[i] = supres.copy()  # coping without addressing the same object
            aux_bars[i].set_index("data", inplace=True)  # setting data as index
            supres = pd.DataFrame(columns=supres.columns)  # reseting pd
        del supres

    elif cfg['MODE'] == 'backtest':
        for j in cfg['BARS']:
            aux_bars[j] = 0
            aux_bars[j] = pd.read_csv("data/wrangling/supresd/"+j+".csv", index_col=0)

    return aux_bars


def initialLoad(cfg):
    if cfg['MODE'] == "training" or cfg['MODE'] == "realtime":
        count = cfg['NCANDLESSAFESTART'] + cfg['NCANDLEST'] + cfg['NCANDLESSAFET'] + 14  # atr

        params = {
            "count": count,
            "granularity": 'M30',
            "price": 'BAM'
        }

        aux_params = {
            "count": (max(cfg['NCANDLESR']) + max(cfg['NCANDLESR']) + 10),
            "granularity": 'D',
            "price": 'M'
        }

        bars = loadBars(cfg['BARS'], cfg, params)
        # i dont want to calculate all supres when i am trading realtime, just to block a signal
        if cfg['MODE'] == "training":
            aux_bars = loadSupres(cfg['AUX_BARS'], cfg, aux_params)
        else:
            aux_bars = bars.copy()
        """ with open ('./real/supres.txt', 'rb') as fp:
            aux_bars = pickle.load(fp)

        with open ('./real/data_test.txt', 'rb') as fp:
            bars = pickle.load(fp) """

    elif cfg['MODE'] == "backtest":
        bars = loadBars(cfg['BARS'], cfg)
        aux_bars = loadSupres(cfg['AUX_BARS'], cfg)

    return bars, aux_bars


def loadTrade(pair, bars_eval, aux_bars, params, strat, cfg):
    pips_factor = 100 if 'JPY' in pair else 1 if 'BTC' in pair or 'LTC' in pair or 'ETH' in pair else 10000

    # calc dos indicadores############
    indicators = stg.calc_indicators(bars_eval[pair], params, strat)

    # corte dos bars iniciais que precisam de load
    bars_aux = aux_bars[pair]  # usado pro supres
    bars_eval[pair] = bars_eval[pair].iloc[cfg['NCANDLESSAFESTART']-1:]

    for key in indicators:
        indicators[key] = indicators[key][cfg['NCANDLESSAFESTART']-1:]

    # conditions trade ############
    [c_buy, c_sell] = stg.condition_trade(bars_eval[pair], indicators, strat)

    buys, sells = {}, {}

    # if i want to trade realtime, the last candle matters
    if cfg['MODE'] == 'realtime':
        for key in c_buy:
            # corte do safetrade
            buys[key] = np.where(c_buy[key])[0]
            sells[key] = np.where(c_sell[key])[0]
    else:
        for key in c_buy:
            # corte do safetrade
            buys[key] = np.where(c_buy[key][:-cfg['NCANDLESSAFET']])[0]
            sells[key] = np.where(c_sell[key][:-cfg['NCANDLESSAFET']])[0]

    bars_temp = bars_eval[pair]
    dates = bars_temp.index.values
    bars_temp = np.array([bars_temp['high_ask'].values, bars_temp['low_ask'].values, bars_temp['close_ask'].values,
                          bars_temp['high_bid'].values, bars_temp['low_bid'].values, bars_temp['close_bid'].values,
                          bars_temp['high'].values, bars_temp['low'].values, bars_temp['close'].values, dates,
                          bars_temp['atr'].values], dtype=object)

    # {'strat': 0, 'v_ent': 0, 'v_saida': 0, 'data_ent': 0, 'data_saida': 0, 'lucro': 0, lastt': 0}
    move = np.array([0, 0, 0, 0, 0, 0, 0], dtype=object)

    buyscand = np.unique(np.concatenate((buys['crossbase'], buys['crossconf']), 0))
    sellscand = np.unique(np.concatenate((sells['crossbase'], sells['crossconf']), 0))

    candt = np.concatenate((buyscand, sellscand), 0)

    return move, candt, bars_aux, pips_factor, bars_temp, buys, sells, indicators,  c_buy, c_sell


def loader(file, cfg):
    while True:
        try:
            with open('data/wrangling/code/'+file, 'rb') as fp:
                archive = pickle.load(fp)
        except Exception as e:
            cfg['LOGGER'].error('erro ao ler o arquivo moving_pairs '+str(e))
        else:
            break

    return archive


def updater(data, currency, file, cfg):
    while True:
        try:
            with open('data/wrangling/code/'+file, 'rb') as fp:
                archive = pickle.load(fp)
        except Exception as e:
            cfg['LOGGER'].error('erro ao ler o arquivo moving_pairs: '+str(e))
        else:
            break

    archive[currency] = data

    with open('data/wrangling/code/'+file, 'wb') as fp:
        pickle.dump(archive, fp)


def saver(data, file):
    with open('data/wrangling/code/'+file, 'wb') as fp:
        pickle.dump(data, fp)
