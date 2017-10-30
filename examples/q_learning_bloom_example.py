#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

__author__ = 'leferrad'

#from rl3.agent.td_lambda import *
from rl3.agent.bloom import *
from rl3.agent.feature import LBPFeatureTransformer
from rl3.environment.base import CubeEnvironment

import matplotlib.pyplot as plt
import numpy as np
import random

# NOTE: It is important to generate a seed only through 'random' module, since it will feed the 'np.random' module

# TODO: sacar el update si el bloom da true y el reward es positivo (i.e. no actualices si no es necesario)
# TODO: se podria tener un modelo aparte que solo se actualice cuando vemos rewards positivos, a modo memoria del buen pasado
# --> Ver como toman replays en DQN

def max_movements(n_iter, max_iters, total_movements=10):
    max_movs_for_random = (n_iter * total_movements) / int(max_iters) + 1
    # max_movs = max_movs_for_random ** 3
    max_movs = int(2.5 ** max_movs_for_random)
    #max_movs = int(2.4270509831248424 ** max_movs_for_random)  # num_aureo * n_caras / n vertices
    max_movs = min(max_movs, 100)
    return max_movs, max_movs_for_random

if __name__ == '__main__':
    reward_function = 'lbph'  # simple or lbph
    seed = random.randint(0, 1000)
    print "Using seed=%i" % seed
    ce = CubeEnvironment(n=3, seed=seed, whiteplastic=False, reward_function=reward_function)
    ce.randomize(1)
    #model = QubeTabularAgent(ce)
    model = QubeBloomPARAgent(ce, LBPFeatureTransformer())
    #model = QubeBloomPARAgent(ce, LBPFeatureTransformer())
    #model = QubeBloomAgent(ce, LBPFeatureTransformer())


    # TODO: crear un mecanismo de attachments para el env (por ej, para monitorear algoritmos seguidos en cada iter)

    gamma = 0.99
    lambda_ = 0.7  # default: 0.7
    N = 10000
    M = 10  # Total movements
    max_movements_for_cur_iter, max_movements_for_random = 0, 0
    totalrewards = np.empty(N)
    for n in range(N):
        #eps = 1.0/np.sqrt(n+1)
        eps = 1.0/(0.001*n+1)
        #eps = 0.7
        prev_max_movements_for_cur_iter, prev_max_movements_for_random = max_movements_for_cur_iter, max_movements_for_random
        max_movements_for_cur_iter, max_movements_for_random = max_movements(n, N, M)
        if prev_max_movements_for_cur_iter != max_movements_for_cur_iter:
            print "Now playing for a max of %i movements..." % max_movements_for_cur_iter
        totalreward = play_one(model, ce, eps, gamma, lambda_=lambda_, max_iters=max_movements_for_cur_iter)  # max_iters=100
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print "Episode:", n, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", \
                totalrewards[max(0, n-100):(n+1)].mean()
        # print "Algorithm followed: %s" % ce.actions_taken
        #ce.render(flat=False)#.savefig("test%02d.png" % m, dpi=865 / c.N)
        seed = random.randint(0, 1000)
        # print "Using seed=%i" % seed
        ce = CubeEnvironment(n=3, seed=seed, whiteplastic=False, reward_function=reward_function)
        if prev_max_movements_for_random != max_movements_for_random:
            print "Now randomizing environment with %i movements..." % max_movements_for_random
        #ce.randomize(max_movements_for_random)
        #ce.randomize(np.random.randint(1, max_movements_for_random+1))
        ce.randomize(random.randint(1, 11))

    print "avg reward for last 100 episodes:", totalrewards[-100:].mean()
    print "total steps:", totalrewards.sum()

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    print "Done!"