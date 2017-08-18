#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

__author__ = 'leferrad'

from rl3.agent.feature import LBPFeatureTransformer
from rl3.agent.td_lambda import QubeRegAgent, QubeTabularAgent, play_one
from rl3.environment.base import CubeEnvironment

import matplotlib.pyplot as plt
import numpy as np


def max_movements(n_iter):
    return int(n_iter / M + 1) * 10

if __name__ == '__main__':
    reward_function = 'lbph'  # simple or lbph
    seed = np.random.randint(0, 1000)
    print "Using seed=%i" % seed
    ce = CubeEnvironment(n=3, seed=seed, whiteplastic=False, reward_function=reward_function)
    ce.randomize(1)
    #model = QubeTabularAgent(ce)
    model = QubeRegAgent(ce, LBPFeatureTransformer())


    # TODO: crear un mecanismo de attachments para el env (por ej, para monitorear algoritmos seguidos en cada iter)

    gamma = 0.99
    lambda_ = 0.7
    N = 5000
    M = N / 10
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        #eps = 1.0/np.sqrt(n+1)
        eps = 1.0/(0.01*n+1)
        #eps = 0.7
        totalreward = play_one(model, ce, eps, gamma, lambda_=lambda_, max_iters=max_movements(n))  # max_iters=100
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print "episode:", n, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", \
                totalrewards[max(0, n-100):(n+1)].mean()
        # print "Algorithm followed: %s" % ce.actions_taken
        #ce.render(flat=False)#.savefig("test%02d.png" % m, dpi=865 / c.N)
        seed = np.random.randint(0, 1000)
        #print "Using seed=%i" % seed
        ce = CubeEnvironment(n=3, seed=seed, whiteplastic=False, reward_function=reward_function)
        ce.randomize(n/M + 1)

    print "avg reward for last 100 episodes:", totalrewards[-100:].mean()
    print "total steps:", totalrewards.sum()

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    print "Done!"