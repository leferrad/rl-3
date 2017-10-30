#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

__author__ = 'leferrad'

from rl3.environment.cube import Cube
from rl3.agent.feature import LBPFeatureTransformer

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


def are_inverse_actions(a1, a2):
    are_inverse = False
    if a1 != a2 and any([a1.replace("'", "") == a2, a2.replace("'", "") == a1]):
        are_inverse = True
    return are_inverse


def simple_reward_2(cube, reward_positive=100, reward_negative=-1):
    reward = reward_negative
    if cube.is_solved():
        reward = reward_positive
    return reward


def simple_reward(cube, reward_positive=100, reward_negative=-1):
    if cube.is_solved():
        reward = reward_positive
    else:
        actions_taken = cube.actions_taken
        # punish to avoid regrets (i.e. agent moving back and forth between two states)
        if len(actions_taken) >= 2 and are_inverse_actions(actions_taken[-2], actions_taken[-1]):
            reward = 10 * reward_negative
        # punish to avoid loops (i.e. agent doing a complete loop up to the same state,
        #                        or making three steps that are equal to make just a single inverse step)
        elif len(actions_taken) >= 3 and len(set(actions_taken[-3:])) == 1:
            reward = 10 * reward_negative
        else:
            reward = reward_negative  # due to take a step
    return reward


def lbph_reward(cube, reward_positive=100, reward_negative=-1):
    if cube.is_solved():
        reward = reward_positive
    else:
        actions_taken = cube.actions_taken
        # punish to avoid regrets (i.e. agent moving back and forth between two states)
        if len(actions_taken) >= 2 and are_inverse_actions(actions_taken[-2], actions_taken[-1]):
            reward = 10 * reward_negative
        # punish to avoid loops (i.e. agent doing a complete loop up to the same state,
        #                        or making three steps that are equal to make just a single inverse step)
        elif len(actions_taken) >= 3 and len(set(actions_taken[-3:])) == 1:
            reward = 10 * reward_negative
        else:
            state = cube.get_state()
            lbp_code = LBPFeatureTransformer.transform(state, normalize=False)
            hist_lbp = LBPFeatureTransformer.hist_lbp_code(lbp_code)
            coefficients = np.linspace(-1.0, 1.0, len(hist_lbp))
            #coefficients[coefficients > 0.0] = 0.0
            reward = sum([c * h for (c, h) in zip(coefficients, hist_lbp)])
            reward += reward_negative  # due to take a step
    return reward


rewards_available = {'simple': simple_reward,
                     'lbph': lbph_reward}


class CubeEnvironment(object):
    def __init__(self, n, seed=123, whiteplastic=False, reward_function='simple'):
        self.cube = Cube(n, seed=seed, whiteplastic=whiteplastic)
        self.actions_taken = []
        self.reward_function = rewards_available[reward_function]
        self.actions_available = actions_available
        self.n_actions = len(actions_available)

    def is_solved(self):
        return self.cube.is_solved()

    def get_state(self):
        return self.cube.get_state()

    def take_action(self, action):
        assert action in actions_available, ValueError("Action '%s' doesn't belong to the supported actions: %s",
                                             str(action), str(actions_available.keys()))
        actions_available[action](self.cube)
        self.actions_taken.append(action)
        reward = self.reward_function(self)
        return self.cube.get_state(), reward, self.is_solved()

    def sample_action(self):
        return np.random.choice(actions_available.keys())

    def render(self, flat=False):
        self.cube.render(flat=flat)

    def randomize_old(self, n=20):
        self.cube.randomize(number=n)

    def randomize(self, n=20):
        for i in range(n):
            action = self.sample_action()
            self.take_action(action)

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
    lbp_code = LBPFeatureTransformer.transform(state)
    print "LBP code: %s" % str(lbp_code)
    lbp_hist = LBPFeatureTransformer.hist_lbp_code(lbp_code)
    print "LBP hist: %s" % str(lbp_hist)
    print "It's solved!" if ce.is_solved() else "Not solved!"
    for a in actions_available:
        print "Taking the following action: %s" % a
        ce.take_action(a)
        state = ce.cube.get_state()
        print "State: %s" % str(state)
        lbp_code = LBPFeatureTransformer.transform(state)
        print "LBP code: %s" % str(lbp_code)
        lbp_hist = LBPFeatureTransformer.hist_lbp_code(lbp_code)
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