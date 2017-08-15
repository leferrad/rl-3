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

def is_inverse_action(a1, a2):
    is_inverse = False
    if a1 != a2 and any([a1.replace("'", "") == a2, a2.replace("'", "") == a1]):
        is_inverse = True
    return is_inverse


def simple_reward_2(cube, reward_positive=50, reward_negative=-1):
    reward = reward_negative
    if cube.is_solved():
        reward = reward_positive
    return reward


def simple_reward(cube, reward_positive=50, reward_negative=-1):
    if cube.is_solved():
        reward = reward_positive
    else:
        actions_taken = cube.actions_taken[-2:]
        if len(actions_taken) == 2 and is_inverse_action(actions_taken[0], actions_taken[1]):
            reward = 10 * reward_negative
        else:
            reward = reward_negative
    return reward


def lbph_reward(cube):
    if cube.is_solved():
        reward = 50
    else:
        actions_taken = cube.actions_taken[-2:]
        if len(actions_taken) == 2 and is_inverse_action(actions_taken[0], actions_taken[1]):
            reward = -10
        else:
            state = cube.get_state()
            lbp_code = LBPFeatureTransformer.transform(state, normalize=False)
            hist_lbp = LBPFeatureTransformer.hist_lbp_code(lbp_code)
            coefficients = [-1.0, 0.0, 0.0, 1.0, 2.0]
            reward = sum([c * h for (c, h) in zip(coefficients, hist_lbp)])
    return reward


rewards_available = {'simple': simple_reward,
                     'lbph': lbph_reward}


def play_one(model, env, eps, gamma, max_iters=1000):
    #env.randomize(20)  # Make 20 random movements to scramble the cube
    observation = env.get_state()

    solved = env.is_solved()
    total_reward = 0
    iters = 0

    while not solved and iters < max_iters:
        # Make a movement
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, solved = env.take_action(action)
        total_reward += reward

        if env.is_solved():
            print "WOW! The cube is solved! Algorithm followed: %s" % str(env.actions_taken)

        # Update the model
        next_state = model.predict(observation)
        # assert(len(next_state.shape) == 1)
        G = reward + gamma*np.max(next_state)
        model.update(prev_observation, action, G)

        iters += 1

    return total_reward


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