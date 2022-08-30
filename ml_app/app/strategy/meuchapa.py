import json

import numpy as np
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.pricing as pricing
from app.utils.general import calcLastSupres, frmt
from app.utils.indicators import ash2, crossed, kijun_sen, spearman
from app.utils.loaders import updater
from oandapyV20.contrib.requests import (MarketOrderRequest, StopLossDetails,
                                         TakeProfitDetails)
from oandapyV20.exceptions import V20Error

# indicators functions and to trade
# funcao para calcular indicadores que pode ser chamada de qualquer arquivo


def tradereal(bars_temp, i, move, direction, bars_aux, strat, supres, cfg, pair):
    bars_temp = bars_temp[:, i]
    atr = bars_temp[10]
    close = bars_temp[8]
    units = 1 if direction == "buy" else -1
    tatic = move[0] + 1

    if not calcLastSupres(units, close, pair, supres, bars_temp[9]):
        cfg['LOGGER'].info("blocked by supres")
        return move

    # types of trade:
    # pullback espera permconf, permbase
    # standard se tiver pelo menos um False nos ultimos 7 candles
    # cont trade por hora é igual ao standard

    ac = accounts.AccountDetails(accountID=cfg['ACCID'])
    rsp = cfg['API'].request(ac)
    balance = float(rsp['account']['balance'])

    # calculo do amount de units pra usd e calculo dos quotes usd, gbp e eur
    cdev = pair[round(len(pair)/2):]
    amntqt = {'USD': [0, 'null'], 'EUR': [0, 'null'], 'GBP': [0, 'null'], 'AUD': [0, 'null'], 'JPY': [0, 'null']}
    for cqt in amntqt:
        if cdev == cqt:
            amntqt[cqt][0] = 1
            amntqt[cqt][1] = pair
        else:
            qtpair = [key for key in cfg['BARS'] if cqt in key and cdev in key]
            amntqt[cqt][1] = qtpair[0]

            # pegar o preco dos amountquotes
            paramsprice = {"instruments": amntqt[cqt][1][:round(len(amntqt[cqt][1])/2)]+'_'+amntqt[cqt][1]
                           [round(len(amntqt[cqt][1])/2):]}
            rprice = pricing.PricingInfo(accountID=cfg['ACCID'], params=paramsprice)
            rp = cfg['API'].request(rprice)
            # setar valor dos amount quotes
            if amntqt[cqt][1][3:] == cqt:
                amntqt[cqt][0] = 1/float(rp['prices'][0]['closeoutBid'])
            else:
                amntqt[cqt][0] = float(rp['prices'][0]['closeoutBid'])

    units *= round((balance*(amntqt['usd'][0])*cfg['RISK'])/(1.5*atr))
    mop = {"instrument": pair[:round(len(pair)/2)]+'_'+pair[round(len(pair)/2):], "units": units}

    drct = 1 if units > 0 else -1

    if cfg['C_PREJATR']:     # stoplosss specified? add it
        slprice = close + ((cfg['C_PREJATR'] * atr) * -drct)
        sl = frmt(slprice, pair)
        mop.update({"stopLossOnFill": StopLossDetails(price=sl).data})

    if cfg['C_LUCATR']:   # takeprofit specified? add it
        tpprice = close + ((cfg['C_LUCATR'] * atr) * drct)
        tp = frmt(tpprice, pair)  # só pra fxcm
        mop.update({"takeProfitOnFill": TakeProfitDetails(price=tp).data})

    """ if self.clargs.takeprofit:   # takeprofit specified? add it
        #tpprice = self.pt[pair]._c[self.pt[pair].idx-1] * (1.0 + (self.clargs.takeprofit/100.0) * drct)
        tpprice = closeprice + ((self.clargs.takeprofit * atr) * drct)
        tp = frmt(tpprice, pair) #só pra fxcm
        mop.update({"takeProfitOnFill": TakeProfitDetails(price=tp).data}) """

    # faço o primeiro trade
    data = MarketOrderRequest(**mop).data
    r = orders.OrderCreate(accountID=cfg['ACCID'], data=data)
    try:
        rsp = cfg['API'].request(r)
    except V20Error as e:
        cfg['LOGGER'].error("v20error: %s", e)
    else:
        cfg['LOGGER'].info("response: %d %s", r.status_code, json.dumps(rsp, indent=2))

        if 'orderFillTransaction' in rsp:

            data = {'atr': atr, 'direction': drct, 'cume': 0, 'tatic': tatic+1}
            updater(data, pair, 'moving_pairs.txt', cfg)

            ordert = rsp['orderFillTransaction']
            # datadb = {'pair': ordert['instrument'].replace("_",""), 'order_id': ordert['tradeopened']['tradeid'],
            #           'date_ent': ordert['time'], 'move': drct, 'pips_ent': ordert['tradeopened']['price'],
            #           'takeprofit': tpprice, 'stoploss': slprice, 'atr': atr, 'risk': cfg['RISK'], 'quoteusd':
            #           amntqt['usd'][0], 'quoteeur': amntqt['eur'][0], 'quotegbp': amntqt['gbp'][0], 'quoteaud':
            #           amntqt['aud'][0], 'quotejpy': amntqt['jpy'][0], 'params': 'ap', 'status': 0}
            # lastid = 1001
            # last_id = insertdata(datadb)
            # baseorderid = ordert['tradeopened']['tradeid']

            move[0] = tatic + 1
            move[1] = close
            move[3] = ordert['time']

    return move


def tradeback(bars_temp, i, move, direction, bars_aux, strat, supres, cfg, pair=False):
    # types of trade:
    # pullback espera permconf, permbase
    # standard se tiver pelo menos um False nos ultimos 7 candles
    # cont trade por hora é igual ao standard

    if direction == 'buy':
        dateb = bars_temp[9][i][0:10]
        try:
            base = bars_aux.index.get_loc(dateb) - 1
        except KeyError:
            j = 1
            idxs = bars_aux.index.values
            while True:
                if bars_temp[9][i-j][0:10] in idxs:
                    dateb = bars_temp[9][i-j][0:10]
                    base = bars_aux.index.get_loc(dateb)
                    break
                j = j+1

        sup = bars_aux.iloc[base, (supres-1)*2]
        if bars_temp[8][i] > sup:
            return move
        # shift para o primeiro candle apos
        for j in range(i+1, len(bars_temp[0])):
            if ((bars_temp[4][j] - bars_temp[8][i]) <= -(bars_temp[10][i] * cfg['C_PREJATR'])):
                # preciso tirar o valor de um suposto take profit parcial
                move[5] = -(bars_temp[10][i] * cfg['C_PREJATR']) - ((bars_temp[2][i] - bars_temp[5][i])/2)
                break

            elif ((bars_temp[3][j] - bars_temp[8][i]) >= bars_temp[10][i] * cfg['C_LUCATR']):
                move[5] = (bars_temp[10][i] * cfg['C_LUCATR']) - ((bars_temp[2][i] - bars_temp[5][i])/2)
                break

            elif j+1 >= len(bars_temp[0]):
                move[5] = bars_temp[8][j]-bars_temp[8][i] - ((bars_temp[2][i] - bars_temp[5][i])/2)
                break
    else:
        dateb = bars_temp[9][i][0:10]
        try:
            base = bars_aux.index.get_loc(dateb) - 2
        except KeyError:
            j = 1
            while True:
                idxs = bars_aux.index.values
                if bars_temp[9][i-j][0:10] in idxs:
                    dateb = bars_temp[9][i-j][0:10]
                    base = bars_aux.index.get_loc(dateb) - 1
                    break
                j = j+1

        res = bars_aux.iloc[base, (supres-1)*2+1]
        if bars_temp[8][i] < res:
            return move
        # shift para o primeiro candle apos
        for j in range(i+1, len(bars_temp[0])):
            if ((bars_temp[8][i] - bars_temp[0][j]) <= -(bars_temp[10][i] * cfg['C_PREJATR'])):
                move[5] = -(bars_temp[10][i] * cfg['C_PREJATR']) - ((bars_temp[2][i] - bars_temp[5][i])/2)
                break

            elif ((bars_temp[8][i] - bars_temp[1][j]) >= bars_temp[10][i] * cfg['C_LUCATR']):
                move[5] = (bars_temp[10][i] * cfg['C_LUCATR']) - ((bars_temp[2][i] - bars_temp[5][i])/2)
                break

            elif j+1 >= len(bars_temp[0]):
                move[5] = bars_temp[8][i]-bars_temp[8][j] - ((bars_temp[2][i] - bars_temp[5][i])/2)
                break
    move[0] = strat
    if direction == 'buy':
        move[1] = bars_temp[2][i]
        move[2] = (move[1] + move[5])  # vlr saida
    else:
        move[1] = bars_temp[5][i]
        move[2] = (move[1] - move[5])  # vlr saida
    move[6] = j  # n de safet
    move[3] = bars_temp[9][i]   # dt entrada
    move[4] = bars_temp[9][j]   # dt saida

    return move


def calc_indicators(bars, params, strat):
    if strat == 'ASH':
        conf1, conf2 = ash2(bars['close'].values, params[0], params[1])
        ks = kijun_sen(bars, params[2], params[2])

    elif strat == 'SPEAR':
        conf1 = spearman(bars['close'].values, int(params[0]))
        conf2 = np.zeros(len(bars['close'].values))
        ks = kijun_sen(bars, params[1], params[1])

    indicators = {'conf1': conf1, 'conf2': conf2, 'baseline': ks}

    # indicators = {'conf': conf}

    return indicators
# //////////////////// fim using_indicators


def condition_trade(bars, indicators, strat):

    permconfbuy = indicators['conf1'] > indicators['conf2']
    crossconfbuy = crossed(indicators['conf1'], indicators['conf2'], direction='above')
    permbasebuy = bars['close'].values > indicators['baseline']
    crossbasebuy = crossed(bars['close'].values, indicators['baseline'], direction='above')

    permconfsell = indicators['conf1'] < indicators['conf2']
    crossconfsell = crossed(indicators['conf1'], indicators['conf2'], direction='below')
    permbasesell = bars['close'].values < indicators['baseline']
    crossbasesell = crossed(bars['close'].values, indicators['baseline'], direction='below')

    condbuy = {'permconf': permconfbuy, 'crossconf': crossconfbuy,
               'permbase': permbasebuy, 'crossbase': crossbasebuy}

    condsell = {'permconf': permconfsell, 'crossconf': crossconfsell,
                'permbase': permbasesell, 'crossbase': crossbasesell}

    return condbuy, condsell
# //////////////////// fim condition_trade


def tatic(buys, sells, bars_temp, indicators, bars_aux, c_buy, c_sell, params, move, i, cfg, checkpullbackb=False,
          checkpullbacks=False, pair=False):
    # cross base
    trade = tradereal if cfg['MODE'] == 'realtime' else tradeback
    direction = ""
    notspike = abs((bars_temp[8][i] - indicators['baseline'][i])) < bars_temp[10][i]

    # there is a pullback to a buy or sell operation?
    if checkpullbackb:
        if notspike and c_buy['permconf'][i] and c_buy['permbase'][i]:
            direction = 'buy'
            move = trade(bars_temp, i, move, direction, bars_aux, 0, params[-1], cfg, pair)
    elif checkpullbacks:
        if notspike and c_sell['permconf'][i] and c_sell['permbase'][i]:
            direction = 'sell'
            move = trade(bars_temp, i, move, direction, bars_aux, 0, params[-1], cfg, pair)

    # i am not checking the pullback, lets check another trade
    else:
        spike = abs((bars_temp[8][i] - indicators['baseline'][i])) > bars_temp[10][i]
        if i in buys['crossbase']:
            direction = 'buy'
            if spike:
                # wait pullback buy
                move[0] = -1
            elif False in c_buy['permconf'][i-7:i] and c_buy['permconf'][i] and notspike:
                # standard
                move = trade(bars_temp, i, move, direction, bars_aux, 1, params[-1], cfg, pair)

        elif i in sells['crossbase']:
            direction = 'sell'
            if spike:
                # wait pullback
                move[0] = -2
            elif False in c_sell['permconf'][i-7:i] and c_sell['permconf'][i] and notspike:
                # standard
                move = trade(bars_temp, i, move, direction, bars_aux, 1, params[-1], cfg, pair)

        # cross conf
        elif i in buys['crossconf'] and i in buys['permbase']:
            direction = 'buy'
            if spike:
                # cont trade
                move = trade(bars_temp, i, move, direction, bars_aux, 2, params[-1], cfg, pair)
            else:
                # standard
                move = trade(bars_temp, i, move, direction, bars_aux, 3, params[-1], cfg, pair)
        elif i in sells['crossconf'] and i in sells['permbase']:
            direction = 'sell'
            if spike:
                # cont trade
                move = trade(bars_temp, i, move, direction, bars_aux, 2, params[-1], cfg, pair)
            else:
                # standard
                move = trade(bars_temp, i, move, direction, bars_aux, 3, params[-1], cfg, pair)

    return move, direction


def calcgain(mvalidation, results, pips_factor, cfg, i):
    mvalidation = (mvalidation * cfg['RISK'] / ((float(results[i, 6])*pips_factor)*cfg['C_PREJATR'])) * \
                  (float(results[i, 5]) * pips_factor) + mvalidation
    return mvalidation


def calcfitness(wrate, n_ops, cfg):
    # ======================= fitness ===============================================
    # w1 = percent if percent > 0 else percent*1.5
    # w1 = w1 * 2 if wrate > 60 else 0
    # w2 = (wrate/55) if wrate > 60 else (wrate /45) if wrate > 70 else -(100/(wrate+0.1))
    w2 = (wrate/55) if wrate > cfg['LIM_WR'] and wrate < 90 else -(100/(wrate+0.1))
    # w3 = sharpe_ratio*5 if w2 > 0 and  sharpe_ratio > 0 else 0
    w2 = w2 + (n_ops/(100) * 5) if wrate > cfg['LIM_WR'] else w2
    # w2 = w2 + (n_ops/(100)) if wrate > 68 else w2
    # w3 = matr_fit

    # w3 = n_ops/1000
    return w2
