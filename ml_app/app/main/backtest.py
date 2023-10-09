import csv
import datetime as dt
from time import time

import numpy as np
import psutil
import ray
from app.optimizers.fitness import fitness_function
from app.optimizers.pso import pso
from app.utils import loaders
from app.utils.general import barscut


def process_backtest(cfg, start):
    if cfg['UPDATE_DATA']:
        loaders.downloadCandles(cfg)
        loaders.downloadSupres(cfg)
    bars, aux_bars = loaders.initialLoad(cfg)
    print('initializing the backtest')
    strat = cfg['STRATEGIES'][0]
    # creating csvs
    num_cpus = psutil.cpu_count(logical=True)
    ray.init(num_cpus=num_cpus*2, num_gpus=10)
    ray_fitness = ray.remote(fitness_function)
    with open('results/monthly_'+strat+'.csv', 'w') as filedata:
        writer = csv.writer(filedata, delimiter=',')
        temp = ['param'+str(j) for j in range(len(cfg['PARAMS_MIN'][strat]))]
        data = ['periodo', 'per acertos', 'per total', 'moves', 'n acertos', 'n erros',
                'open', 'close', 'high', 'low', 'fitnesst', 'wint']
        data.extend(temp)
        del temp
        writer.writerow(data)

    with open('results/allmoves_'+strat+'.csv', 'w') as filedata:
        writer = csv.writer(filedata, delimiter=',')
        data = ['periodo', 'entrada', 'saida', 'acao', 'v entrada', 'v saida', 'lucro',  'atr', 'strat', 'currency']
        writer.writerow(data)

    per = 1
    earnings = cfg['EARNINGS']
    # iteradores dos cut
    k = 0
    while True:
        if per > 120:
            d_test, d_validation = {}, {}

            # corte dos dados de teste e validacao
            # eu corto a validação em: nteste - safestart até nteste + validacao + safestart
            k = (per-1)*cfg['NCANDLESV']
            bars_temp = bars.copy()
            d_test = barscut(bars_temp, k, cfg)
            d_validation = barscut(bars_temp, (k+cfg['NCANDLEST']-cfg['NCANDLESSAFESTART']), cfg, train=False)

            """#test de realtime
            for i in cfg['BARS']:
                d_validation[i] = bars_temp[i][-472:]
                d_test[i] = bars_temp[i][-2016-188:-188] """

            # defino as datas de ano e mes da validação, essas datas tbm serão usadas para checar se é o fim do algo
            index = next(iter(d_validation))
            dates = dt.datetime.strptime(d_validation[index].index[cfg['NCANDLESSAFESTART']], "%Y-%m-%d %H:%M:%S")
            print('period '+str(per)+'. validation starting from '+str(dates))
            del index
            del dates

            print(f"\033[1;31m, Initiating the optimizer {time()-start:.4f}s \033[1;37m")
            best = np.array([-1000 for _ in range(cfg['PARAM_FIT'])])

            for pair in aux_bars:
                for r in range(0, cfg['NRUNS']):
                    # call do optimizer
                    b_temp, s_temp = pso(d_test[pair], cfg['PARAMS_MAX'][strat], cfg['PARAMS_MIN'][strat],
                                        strat, best, aux_bars[pair], cfg, start, r, ray_fitness)

                    print(f"\033[1;31m to optimize {time()-start:.4f}s \033[1;37m")
                    print('--------------')

                    if b_temp[0] > best[0]:
                        best = b_temp
                        # sol = s_temp
                    # fim dos runs!
                best = list(best)

                candidate = fitness_function(d_validation, aux_bars, best[5], strat, earnings,
                                            per, cfg, best[1], best[0], validation=True)

            earnings = candidate[6]

        per = per + 1
