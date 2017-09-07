#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

__author__ = 'leferrad'

import numpy as np
import matplotlib.pyplot as plt


def plot_moving_avg(total_rewards, show=True):
    # See https://github.com/lazyprogrammer/machine_learning_examples/blob/master/rl2/cartpole/q_learning_bins.py
    N = len(total_rewards)
    moving_avg = np.empty(N)
    for t in range(N):
        moving_avg[t] = total_rewards[max(0, t-100):(t+1)].mean()
    plt.plot(moving_avg)
    plt.title("Total rewards - moving average")
    if show:
        plt.show()
