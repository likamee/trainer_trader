import collections
import random
from time import time

import numpy as np
import ray

# from app.optimizers.fitness import fitness_function


def pso(data, params_max, params_min, strat, best, aux_bars, cfg, start,
        r=1, ray_fitness=False, per=False, earnings=False):
    qnt_p = len(params_min)
    n_ind = cfg['N_IND'][cfg['STRATEGIES'].index(strat)]
    sol_vector = np.array([np.array([0] * qnt_p) for _ in range(n_ind)])
    for i in range(qnt_p):
        sol_vector[:, i] = [random.randint(params_min[i], params_max[i]) for p in range(n_ind)]

    hist_sol = sol_vector[:]
    h_gbest_fit = []
    params_fit = np.array([float(-10000) for _ in range(cfg['PARAM_FIT'])])
    pbest = np.array([params_fit for _ in range(n_ind)], dtype=object)
    gbest = np.array([float(-10000) for _ in range(cfg['PARAM_FIT'])], dtype=object)

    vel_arr = ([np.array([0] * qnt_p) for _ in range(n_ind)])
    it = 0

    while it < cfg['ITERS']:
        # ray remote

        candidate = ray.get([ray_fitness.remote(data, aux_bars, sol_vector[i], strat, earnings, per, cfg)
                            for i in range(n_ind)])
        # candidate = np.zeros(n_ind, dtype=object)

        for i in range(n_ind):
            # candidate[i] = fitness_function(data, aux_bars, sol_vector[i], strat, earnings, per, cfg)
            if(pbest[i][0] < candidate[i][0]):
                pbest[i] = candidate[i]

            if(gbest[0] < candidate[i][0]):
                gbest = candidate[i]

        print(f"\033[1;31mindicators {time()-start:.4f}s \033[1;37m")
        print('--------------')

        for i in range(n_ind):
            r1 = np.array([random.uniform(0, 1)] * qnt_p)
            r2 = np.array([random.uniform(0, 1)] * qnt_p)

            vel_arr[i] = (cfg['W']*vel_arr[i]) + (np.array([cfg['C1']] * qnt_p)) * \
                         (r1 * (pbest[i][5] - sol_vector[i])) + (np.array([cfg['C2']] * qnt_p)) * \
                         (r2 * (gbest[5]-sol_vector[i]))

            sol_vector[i] = vel_arr[i] + sol_vector[i]

            sol_vector[i][:] = sol_vector[i][:].astype(int)

            for j in range(qnt_p):
                sol_vector[i][j] = sol_vector[i][j] if sol_vector[i][j] < params_max[j] and \
                                   sol_vector[i][j] > params_min[j] else params_max[j] if \
                                   sol_vector[i][j] > params_max[j] else params_min[j]

            cont = 0

            if strat == 'ash':
                while (hist_sol == sol_vector[i]).all(1).any() or \
                      (hist_sol == sol_vector[i][[1, 0, 2, 3]]).all(1).any():
                    ind = random.randint(0, len(sol_vector[0])-1)
                    sol_vector[i][ind] = random.randint(params_min[ind], params_max[ind])
                    cont = cont + 1
                    if cont > 50:
                        break
            else:
                while (hist_sol == sol_vector[i]).all(1).any():
                    ind = random.randint(0, len(sol_vector[0])-1)
                    sol_vector[i][ind] = random.randint(params_min[ind], params_max[ind])
                    cont = cont + 1
                    if cont > 50:
                        break
            hist_sol = np.vstack((hist_sol, sol_vector[i]))

        if r == 0:
            best = gbest

        print('-')
        print("best sol:", gbest[5],
              " current iter: ", it, ' fitness:', "{:.2f}".format(gbest[0]))
        print("amount: ", "{:.2f}".format(gbest[3]), " per: ", "{:.2f}".format(gbest[4]),
              " wr:", "{:.2f}".format(gbest[1]), ' n op:', gbest[2], ' run:', r)
        print("r best sol:", best[5], ' r fitness:', "{:.2f}".format(best[0]), " r amount: ", "{:.2f}".format(best[3]))
        print('-')
        h_gbest_fit.append(gbest)
        fit_over_it = [row[0] for row in h_gbest_fit]
        repeats = max(collections.Counter(fit_over_it).values())

        if repeats > cfg['IMAXREPEATS']:
            print('muitas repeticoes, saindo...')
            break

        it = it + 1

    print("the best position is ", gbest[5], "in iteration number ", it)
    return gbest, gbest[5]
