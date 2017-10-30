#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

__author__ = 'leferrad'


import matplotlib
matplotlib.use("Agg")


from rl3.agent.ac.elm import *
from rl3.agent.feature import LBPFeatureTransformer
from rl3.environment.base import CubeEnvironment
from rl3.util.plot import plot_moving_avg


import matplotlib.pyplot as plt
import numpy as np
import random

# NOTE: It is important to generate a seed only through 'random' module, since it will feed the 'np.random' module


def max_movements(n_iter, max_iters, total_movements=10):
    max_movs_for_random = (n_iter * total_movements) / int(max_iters) + 1
    # max_movs = max_movs_for_random ** 3
    max_movs = int(2.5 ** max_movs_for_random)
    #max_movs = int(2.4270509831248424 ** max_movs_for_random)  # num_aureo * n_caras / n vertices
    max_movs = min(max_movs, 50)
    return max_movs, max_movs_for_random


def get_max_movements_probabilistic(max_movs):
    movements = range(1, max_movs+1)
    sum_movements = sum(movements)
    probabilities = [m/float(sum_movements) for m in movements]
    return int(np.random.choice(movements, p=probabilities))

if __name__ == '__main__':
    reward_function = 'simple'  # simple or lbph
    seed = random.randint(0, 1000)
    print "Using seed=%i" % seed
    ce = CubeEnvironment(n=3, seed=seed, whiteplastic=False, reward_function=reward_function)
    ce.randomize(1)

    # Create actor-critic models
    n_hidden = 50
    activation_func = 'gaussian'
    t = 10
    gamma = 0.99
    lambda_ = 0.3  # default: 0.7
    N = 5000
    M = 8  # Total movements

    policy_model = PolicyModel(ce, LBPFeatureTransformer(), t=t, lambda_=lambda_, gamma=gamma)
    value_model = ValueModel(ce, LBPFeatureTransformer(), n_hidden=n_hidden, activation_func=activation_func,
                             t=t, lambda_=lambda_, gamma=gamma)


    max_movements_for_cur_iter, max_movements_for_random = 0, 0
    total_rewards = np.empty(N)
    total_iters = np.empty(N)
    for n in range(N):
        #eps = 1.0/np.sqrt(n+1)
        eps = 1.0/(0.001*n+1)
        #eps = 0.7
        prev_max_movements_for_cur_iter, prev_max_movements_for_random = max_movements_for_cur_iter, max_movements_for_random
        max_movements_for_cur_iter, max_movements_for_random = max_movements(n, N, M)
        if prev_max_movements_for_cur_iter != max_movements_for_cur_iter:
            print "Now playing for a max of %i movements..." % max_movements_for_cur_iter
        total_reward, iters = play_one(policy_model, value_model, ce,
                                       max_iters=max_movements_for_cur_iter)  # max_iters=100
        total_rewards[n] = total_reward
        total_iters[n] = iters

        if n % 100 == 0:
            print "Episode:", n, "total reward:", total_reward, "avg reward (last 100):", \
                total_rewards[max(0, n-100):(n+1)].mean()
        #ce.render(flat=False)#.savefig("test%02d.png" % m, dpi=865 / c.N)
        seed = random.randint(0, 1000)
        ce = CubeEnvironment(n=3, seed=seed, whiteplastic=False, reward_function=reward_function)
        if prev_max_movements_for_random != max_movements_for_random:
            print "Now randomizing environment with a max of %i movements..." % max_movements_for_random
        #ce.randomize(max_movements_for_random)
        #ce.randomize(random.randint(1, max_movements_for_random+1))
        ce.randomize(get_max_movements_probabilistic(max_movements_for_random))

    print "avg reward for last 100 episodes:", total_rewards[-100:].mean()
    print "total steps:", total_rewards.sum()

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_moving_avg(total_rewards, title="Total rewards per episode- moving average")
    plot_moving_avg(total_iters, title="Total iterations per episode - moving average")

    print "Done!"