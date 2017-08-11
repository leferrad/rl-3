#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

__author__ = 'leferrad'

from rl3.environment.cube import Cube
from rl3.agent.feature import FeatureTransformer

import matplotlib.pyplot as plt
import numpy as np

actions_available = {"U": lambda c: c.move("U", 0, 1),
                     "U'": lambda c: c.move("U", 0, -1),
                     "D": lambda c: c.move("D", 0, 1),
                     "D'": lambda c: c.move("D", 0, -1),
                     "L": lambda c: c.move("L", 0, 1),
                     "L'": lambda c: c.move("L", 0, -1),
                     "R": lambda c: c.move("R", 0, 1),
                     "R'": lambda c: c.move("R", 0, -1),
                     "F": lambda c: c.move("F", 0, 1),
                     "F'": lambda c: c.move("F", 0, -1),
                     "B": lambda c: c.move("B", 0, 1),
                     "B'": lambda c: c.move("B", 0, -1)}


def simple_reward(cube, reward_positive=50, reward_negative=-1):
    reward = reward_negative
    if cube.is_solved() is True:
        reward = reward_positive
    return reward

rewards_available = {'simple': simple_reward,
                     }


def play_one(model, env, eps, gamma, max_iters=1000):
    env.randomize(20)  # Make 20 random movements to scramble the cube
    observation = env.get_state()

    solved = False
    total_reward = 0
    iters = 0

    while not solved and iters < max_iters:
        # Make a movement
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, solved = env.take_action(action)

        total_reward += reward

        # Update the model
        model.update(prev_observation, action, reward)

        iters += 1

    return total_reward


class CubeEnvironment(object):
    def __init__(self, n, seed=123, whiteplastic=False, reward_function='simple'):
        self.cube = Cube(n, seed=seed, whiteplastic=whiteplastic)
        self.actions_taken = []
        self.reward_function = rewards_available[reward_function]

    def is_solved(self):
        return self.cube.is_solved()

    def take_action(self, action):
        assert action in actions_available, ValueError("Action '%s' doesn't belong to the supported actions: %s",
                                             str(action), str(actions_available.keys()))
        actions_available[action](self.cube)
        self.actions_taken.append(action)
        reward = self.reward_function(self.cube)
        return self.cube.get_state(), reward, self.is_solved()

    def render(self, flat=False):
        self.cube.render(flat=flat)

    def randomize(self, n=20):
        self.cube.randomize(number=n)

if __name__ == "__main__":
    """
    Functional testing.
    """
    # Taking all of the supported actions

    seed = np.random.randint(0, 100)
    print "Using seed=%i" % seed
    ce = CubeEnvironment(n=3, seed=seed, whiteplastic=False)
    print "---o- ---------------------------- -o---"
    print "---o- Taking all supported actions -o---"
    print "---o- ---------------------------- -o---"
    state = ce.cube.get_state()
    print "State: %s" % str(state)
    lbp_code = FeatureTransformer.lbp_cube(state)
    print "LBP code: %s" % str(lbp_code)
    lbp_hist = FeatureTransformer.hist_lbp_code(lbp_code)
    print "LBP hist: %s" % str(lbp_hist)
    print "It's solved!" if ce.is_solved() else "Not solved!"
    for a in actions_available:
        print "Taking the following action: %s" % a
        ce.take_action(a)
        state = ce.cube.get_state()
        print "State: %s" % str(state)
        lbp_code = FeatureTransformer.lbp_cube(state)
        print "LBP code: %s" % str(lbp_code)
        lbp_hist = FeatureTransformer.hist_lbp_code(lbp_code)
        print "LBP hist: %s" % str(lbp_hist)
        print "It's solved!" if ce.is_solved() else "Not solved!"
        #ce.render(flat=False)#.savefig("test%02d.png" % m, dpi=865 / c.N)

    print "Algorithm followed: %s" % ce.actions_taken

    # Now, let's take some random actions

    n_random_actions = 10
    print "---o- ------------------------ -o---"
    print "---o- Taking %i random actions -o---" % n_random_actions
    print "---o- ------------------------ -o---"
    ce = CubeEnvironment(n=3, seed=seed, whiteplastic=False)

    for m in range(n_random_actions):
        print "State: %s" % str(ce.cube.get_state())
        print "It's solved!" if ce.is_solved() else "Not solved!"
        #ce.render(flat=False)#.savefig("test%02d.png" % m, dpi=865 / c.N)
        action = np.random.choice(actions_available.keys())
        print "Taking the following action: %s" % action
        ce.take_action(action)

    print "Algorithm followed: %s" % ce.actions_taken

    plt.show()