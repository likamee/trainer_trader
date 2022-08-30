
import json
import logging

import numpy as np
import psutil
import ray
from app.optimizers.fitness import fitness_function
from app.optimizers.pso import pso
from app.utils import loaders


def process_training(cfg, start):
    num_cpus = psutil.cpu_count(logical=True)
    ray.init(num_cpus=num_cpus)
    ray_fitness = ray.remote(fitness_function)
    bars, aux_bars = loaders.initialLoad(cfg)

    logging.basicConfig(
            filename="data/wrangling/code/crypto.log",
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s : %(message)s',
    )

    cfg['LOGGER'] = logging.getLogger(__name__)
    print('initializing the infinite training')

    d_test = bars

    # call do optimizer
    for strat in cfg['STRATEGIES']:
        best = np.array([-1000 for _ in range(cfg['PARAM_FIT'])])
        for r in range(cfg['NRUNS']):
            b_temp, s_temp = pso(d_test, cfg['PARAMS_MAX'][strat], cfg['PARAMS_MIN'][strat], strat,
                                 best, aux_bars, cfg, start, r, ray_fitness)

            if b_temp[0] > best[0]:
                best = b_temp
                sol = s_temp

        sol = dict(enumerate(sol.flatten(), 1))
        sol = {str(k): str(v) for k, v in sol.items()}
        data = {'params': json.dumps(sol), 'wrt': best[1], 'n_ops': best[2]}

        loaders.saver(data, 'params_'+strat+'.pickle')
    print('train has finished!')
