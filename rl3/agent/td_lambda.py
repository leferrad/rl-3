#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

__author__ = 'leferrad'

from rl3.agent.feature import LBPFeatureTransformer

from sklearn.linear_model import PassiveAggressiveRegressor
from bloom_filter import BloomFilter
import numpy as np


class QubeTabularAgent(object):
    def __init__(self, env):
        self.env = env
        self.models = {}
        self.feature_transformer = LBPFeatureTransformer()
        for a in env.actions_available:
            self.models[a] = dict()

    def predict_from_action(self, s, a):
        X = self.feature_transformer.transform(s, normalize=False)
        X = tuple(X)
        prediction = 0.0
        if X in self.models[a]:
            prediction = self.models[a][X]
        return prediction

    def predict(self, s):
        return np.array([self.predict_from_action(s, a) for a in self.models.keys()])

    def update(self, s, a, G):
        X = self.feature_transformer.transform(s, normalize=False)
        X = tuple(X)
        if X in self.models[a]:
            self.models[a][X] += G
        else:
            self.models[a][X] = np.random.uniform(0.0, 1.0)

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.sample_action()
        else:
            actions = self.models.keys()
            G = [self.predict_from_action(s, a) for a in self.models.keys()]
            if max(G) == 0.0:
                action = np.random.choice(actions)
            else:
                action = actions[np.argmax(G)]
            return action


class SGDRegressor:
    def __init__(self, D, lr=1e-4):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = lr

    def partial_fit(self, x, y, e):
        if isinstance(x, np.ndarray) is False:
            x = np.array(x)
        if isinstance(y, np.ndarray) is False:
            y = np.array(y)
        self.w += self.lr*(y - x.dot(self.w))*e

    def predict(self, x):
        x = np.array(x)
        return x.dot(self.w)


class QubeRegAgent(object):
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = {}
        self.feature_transformer = feature_transformer
        for a in env.actions_available:
            self.models[a] = SGDRegressor(feature_transformer.dimensions)
        self.eligibilities = np.zeros((env.n_actions, feature_transformer.dimensions))

    def reset(self):
        self.eligibilities = np.zeros_like(self.eligibilities)

    def predict_from_action(self, s, a):
        X = self.feature_transformer.transform(s, normalize=True)
        return self.models[a].predict(X)

    def predict(self, s):
        return np.array([self.predict_from_action(s, a) for a in self.models.keys()])

    def update(self, s, a, G, gamma=0.99, lambda_=0.7):
        X = self.feature_transformer.transform(s, normalize=True)

        # slower
        # for action in range(self.env.action_space.n):
        #   if action != a:
        #     self.eligibilities[action] *= gamma*lambda_
        #   else:
        #     self.eligibilities[a] = grad + gamma*lambda_*self.eligibilities[a]

        self.eligibilities *= gamma*lambda_
        index_a = self.env.actions_available.keys().index(a)
        self.eligibilities[index_a] += X
        self.models[a].partial_fit(X, [G], self.eligibilities[index_a])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.sample_action()
        else:
            actions = self.models.keys()
            G = [self.predict_from_action(s, a) for a in self.models.keys()]
            return actions[np.argmax(G)]

class QubeBloomAgent(object):
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = {}
        self.feature_transformer = feature_transformer
        for a in env.actions_available:
            self.models[a] = SGDRegressor(feature_transformer.dimensions)
        self.eligibilities = np.zeros((env.n_actions, feature_transformer.dimensions))
        self.bloom_states = BloomFilter(max_elements=256**2)

    def reset(self):
        self.eligibilities = np.zeros_like(self.eligibilities)

    def predict_from_action(self, s, a):
        X = self.feature_transformer.transform(s, normalize=True)
        return self.models[a].predict(X)

    def predict(self, s):
        return np.array([self.predict_from_action(s, a) for a in self.models.keys()])

    def update(self, s, a, G, gamma=0.99, lambda_=0.7):
        X = self.feature_transformer.transform(s, normalize=True)

        # slower
        # for action in range(self.env.action_space.n):
        #   if action != a:
        #     self.eligibilities[action] *= gamma*lambda_
        #   else:
        #     self.eligibilities[a] = grad + gamma*lambda_*self.eligibilities[a]

        self.eligibilities *= gamma*lambda_
        index_a = self.env.actions_available.keys().index(a)
        self.eligibilities[index_a] += X
        self.models[a].partial_fit(X, [G], self.eligibilities[index_a])

    def sample_action(self, s, eps):
        x = tuple(self.feature_transformer.transform(s, normalize=False))
        if x in self.bloom_states:
            # Maybe it's a seen state
            actions = self.models.keys()
            G = [self.predict_from_action(s, a) for a in self.models.keys()]
            if np.max(G) > 0.0:
                # It's a good state to exploit
                self.bloom_states.add(x)
                return actions[np.argmax(G)]
            else:
                # It's not important, so we can explore
                self.bloom_states.add(x)
                return self.env.sample_action()
        else:
            # It's not a seen state, so we can explore
            self.bloom_states.add(x)
            return self.env.sample_action()


class QubePARAgent(object):
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = {}
        self.feature_transformer = feature_transformer
        for a in env.actions_available:
            self.models[a] = PassiveAggressiveRegressor(C=1.0, fit_intercept=True, n_iter=10)
        self.eligibilities = np.zeros((env.n_actions, feature_transformer.dimensions))

    def reset(self):
        self.eligibilities = np.zeros_like(self.eligibilities)

    def predict(self, s):
        return np.array([self.predict_from_action(s, a) for a in self.models.keys()])

    def predict_from_action(self, s, a):
        X = self.feature_transformer.transform(s, normalize=True)
        return self.models[a].predict(X)

    def update(self, s, a, G, gamma=0.99, lambda_=0.7):
        X = self.feature_transformer.transform(s, normalize=True)

        # slower
        # for action in range(self.env.action_space.n):
        #   if action != a:
        #     self.eligibilities[action] *= gamma*lambda_
        #   else:
        #     self.eligibilities[a] = grad + gamma*lambda_*self.eligibilities[a]

        self.eligibilities *= gamma*lambda_
        self.eligibilities[a] += X
        self.models[a].partial_fit(X, [G], self.eligibilities[a])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.sample_action()
        else:
            actions = self.models.keys()
            G = [self.predict_from_action(s, a) for a in self.models.keys()]
            return actions[np.argmax(G)]


def play_one(model, env, eps, gamma=0.99, lambda_=0.7, max_iters=1000):
    env.actions_taken = []  # Reset actions taken on the scramble stage
    observation = env.get_state()

    total_reward = 0
    iters = 0

    while not env.is_solved() and iters < max_iters:
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
        model.update(prev_observation, action, G, gamma, lambda_)

        iters += 1

    return total_reward