#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

__author__ = 'leferrad'

from rl3.agent.feature import LBPFeatureTransformer
from rl3.agent.q_learning import QubeRegAgent, QubeTabularAgent
from rl3.environment.base import CubeEnvironment, play_one

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    reward_function = 'lbph'  # simple or lbph
    seed = 39# np.random.randint(0, 100)
    print "Using seed=%i" % seed
    ce = CubeEnvironment(n=3, seed=seed, whiteplastic=False, reward_function=reward_function)
    ce.randomize(1)
    model = QubeTabularAgent(ce)
    #model = QubeRegAgent(ce, LBPFeatureTransformer())


    # TODO: crear un mecanismo de attachments para el env (por ej, para monitorear algoritmos seguidos en cada iter)

    gamma = 0.99
    N = 3000
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        eps = 1.0/np.sqrt(n+1)
        totalreward = play_one(model, ce, eps, gamma, max_iters=100)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print "episode:", n, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", \
                totalrewards[max(0, n-100):(n+1)].mean()
        # print "Algorithm followed: %s" % ce.actions_taken
        #ce.render(flat=False)#.savefig("test%02d.png" % m, dpi=865 / c.N)
        seed = np.random.randint(0, 100)
        #print "Using seed=%i" % seed
        ce = CubeEnvironment(n=3, seed=seed, whiteplastic=False, reward_function=reward_function)
        ce.randomize(n/100 + 1)

    print "avg reward for last 100 episodes:", totalrewards[-100:].mean()
    print "total steps:", totalrewards.sum()

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    print "Done!"