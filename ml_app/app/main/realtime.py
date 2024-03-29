import datetime as dt
import json
import logging
import time

import app.strategy.meuchapa as stg
from app.main import training
from app.utils.general import checkmovingpairs, genmovingpairs
from app.utils.loaders import initialLoad, loader, loadTrade
from pytz import timezone


def process_realtime(cfg, start):
    logging.basicConfig(
            filename="data/wrangling/code/crypto.log",
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s : %(message)s',
    )

    cfg['LOGGER'] = logging.getLogger(__name__)

    cfg['MODE'] = 'training'
    training.process_training(cfg, start)
    cfg['MODE'] = 'realtime'
    # bkhours = np.array([['2020','2250']])
    params = {}
    for strat in cfg['STRATEGIES']:
        params[strat] = loader('params_'+strat+'.pickle', cfg)['params']
    movingpairs = genmovingpairs(cfg)
    checkpullbackb, checkpullbacks = {}, {}
    for strat in cfg['STRATEGIES']:
        checkpullbackb[strat], checkpullbacks[strat] = {}, {}
        for pair in cfg['BARS']:
            checkpullbackb[strat][pair] = False
            checkpullbacks[strat][pair] = False

    print('initiated the listening!')
    now = dt.datetime.now()
    cfg['LOGGER'].info("algo initiated at: %s:%s:%s", now.hour, now.minute, now.second)
    trained = True
    while True:
        now = dt.datetime.now()
        now = now.astimezone(timezone('America/Sao_Paulo'))
        if now.minute % 30 == 0 and (now.hour > 19 or now.hour < 17):
            trained = False
            movingpairs = loader('moving_pairs.txt', cfg)
            cfg['LOGGER'].info("Searching for operations")
            bars, aux_bars = initialLoad(cfg)
            pairs = cfg['PAIRS']
            for pair in pairs:
                if movingpairs[pair]['tatic'] != 0:
                    movingpairs[pair] = checkmovingpairs(pair, movingpairs, cfg)
                    if movingpairs[pair]['tatic'] != 0:
                        cfg['LOGGER'].info("there is an operation in course for this pair")
                        continue

                for strat in cfg['STRATEGIES']:
                    params[strat] = loader('params_'+strat+'.pickle', cfg)['params']
                    params[strat] = json.loads(params[strat])
                    # convert to list of ints the dict of string generated by json.loads
                    params[strat] = list(map(int, list(params[strat].values())))
                    move, candt, bars_aux, pips_factor, bars_temp, buys, sells, \
                        indicators, c_buy, c_sell = loadTrade(pair, bars.copy(), aux_bars, params[strat], strat, cfg)

                    # the candt array has any trade in the last pos? (now = -2)
                    i = len(bars_temp[0]) - 2
                    if i in candt and (((bars_temp[2][i] - bars_temp[5][i]) * pips_factor < cfg['SPREAD_CUT'])
                                       and (bars_temp[10][i] * pips_factor) > cfg['ATR_CUT']):
                        move, direction = stg.tatic(buys, sells, bars_temp, indicators, bars_aux, c_buy, c_sell,
                                                    params[strat], move, i, cfg, checkpullbackb[strat][pair],
                                                    checkpullbacks[strat][pair], pair)
                        cfg['LOGGER'].info("%s - processing: direction: %s. state: %s.  strategy: %s",
                                           pair, direction, move[0], strat)

                    checkpullbackb[strat][pair], checkpullbacks[strat][pair] = False, False

                    if move[1] == 0:
                        if move[0] == -1:
                            checkpullbackb[strat][pair] = True
                        elif move[0] == -2:
                            checkpullbacks[strat][pair] = True

                        continue
                    else:
                        break

        elif now.hour > 17 and now.hour < 19 and trained is False:
            cfg['MODE'] = 'training'
            cfg['LOGGER'].info("starting the training")
            training.process_training(cfg, start)
            cfg['LOGGER'].info("finished the training")
            cfg['MODE'] = 'realtime'
            trained = True
        time.sleep(60)
