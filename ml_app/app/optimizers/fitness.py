import csv

import app.strategy.meuchapa as stg
import app.utils.loaders as utils
import numpy as np


def on_evaluate(params, bars_eval, aux_bars, strat, cfg, validation=False):

    results = []
    bars_eval = bars_eval.copy()

    for pair in cfg['PAIRS']:

        move, candt, bars_aux, pips_factor, bars_temp, \
         buys, sells, indicators, c_buy, c_sell = utils.loadTrade(pair, bars_eval, aux_bars, params, strat, cfg)

        results_temp = []
        idxjump = 0

        for i in candt:
            checkpullbackb, checkpullbacks = False, False
            if move[0] == -1:
                checkpullbackb = True
            elif move[0] == -2:
                checkpullbacks = True
            move.fill(0)

            if (i + 2) >= len(bars_temp[0]) or i < idxjump or i == 0 or ((bars_temp[2][i] - bars_temp[5][i]) *
               pips_factor > cfg['SPREAD_CUT']) or (bars_temp[10][i] * pips_factor) < cfg['ATR_CUT']:
                continue

            move, direction = stg.tatic(buys, sells, bars_temp, indicators, bars_aux, c_buy, c_sell, params, move, i,
                                        cfg, checkpullbackb, checkpullbacks)

            # i am not trading this candle
            if move[1] == 0:
                continue
            else:
                idxjump = move[6]

            atr = bars_temp[10][i] if move[0] != 0 else bars_temp[10][i+1]
            # dt entrada, dt saida, dir, vlr entrada, vlr saida, lucro atr, atr, lucro ind, strat, dt saida parcial
            # data = np.array([move[3], move[4], direction, move[1], move[2],
            #                  move[5], atr, move[6], move[0], move[7]], dtype=object)

            # dt entrada, dt saida, dir, vlr entrada, vlr saida, lucro, atr, strat, pair, lastt
            # stop loss direto
            data = np.array([move[3], move[4], direction, move[1], move[2], move[5], atr, move[0], pair], dtype=object)

            results_temp.append(data)

        results.append(results_temp)

    return results


# //////////////////// fim on_evaluate
def fitness_function(bars_temp, aux_bars, params, strat, earnings, per, cfg,
                     params_wr=False, fit_train=False, validation=False):
    results = []
    # 0 open - 1 close - 2 high - 3 low
    mvalidation = minicial = high = low = percent = 10000 if earnings is False else earnings[1]
    # start = time.time()
    # pego os candles de teste ou validacao pra fazer o evaluate
    results = on_evaluate(params, bars_temp, aux_bars, strat, cfg, validation)

    results = np.asarray(results)

    stacked = np.array([])
    for i in range(len(results)):
        stacked = np.vstack([stacked, np.array(results[i])]) if \
                  np.array(results[i]).size and stacked.size else \
                  np.array(results[i]) if np.array(results[i]).size else \
                  stacked

    # caso nao houve negociacao no periodo
    if len(stacked) == 0:
        fitness = wrate = n_ops = 0
        ret = [fitness, wrate, n_ops, mvalidation, percent, params, earnings]
        return ret

    results = stacked

    # verificacao das metricas de winrate
    target = results[:, 5].astype(float)
    # total
    n_ops = len(target)
    # negativos
    negatives = sum(1 for i in target if i < 0)
    # positives
    positives = sum(1 for i in target if i > 0)
    # pernegatives
    wrate = positives/len(target)*100 if n_ops > 0 else 0
    # acerto_total = (positives/total)*100

    for i in range(len(results)):
        pips_factor = 100 if 'JPY' in results[i, 8] else 1 if 'BTC' or 'LTC' or 'ETH' in results[i, 8] else 10000

        mvalidation = stg.calcgain(mvalidation, results, pips_factor, cfg, i)

        low = mvalidation if mvalidation < low else low
        high = mvalidation if mvalidation > high else high

        # row ->> #per, dt entrada, dt saida, dir, vlr entrada, vlr saida, lucro, atr, strat
        # allmoves ['periodo', 'currency', 'entrada', 'saida', 'acao', 'v entrada',
        #           'v saida', 'lucro',  'atr', 'strat', 'amount']

    percent = (mvalidation - minicial) / minicial

    # save only in the backtest (validation)
    if validation:
        qnt_p = len(cfg['PARAMS_MIN'][strat])
        col = np.array([per] * len(results))
        col.shape = (len(results), 1)
        results = np.hstack((col, results))
        f = open('results/allmoves_'+strat+'.csv', 'a')
        np.savetxt(f, results, '%s', ',')
        f.close()

        fitness = 0
        earnings = [minicial, mvalidation, high, low]

        with open('results/monthly_'+strat+'.csv', 'a') as filedata:
            writer = csv.writer(filedata, delimiter=',')
            temp = [str(params[j]) for j in range(qnt_p)]
            data = [str(per), wrate, percent, n_ops, positives, negatives, str(earnings[0]), str(earnings[1]),
                    str(earnings[2]), str(earnings[3]), fit_train, params_wr]
            data.extend(temp)
            del temp
            writer.writerow(data)

    fitness = stg.calcfitness(wrate, n_ops, cfg)
    ret = [fitness, wrate, n_ops, mvalidation, percent, params, earnings]
    return ret
# //////////////////// fim fitness_function
