#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

__author__ = 'leferrad'

from sklearn.linear_model import PassiveAggressiveRegressor, SGDRegressor
from bloom_filter import BloomFilter
import numpy as np


class SGDRegressor2:
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


class QubeBloomRegAgent(object):
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = {}
        self.feature_transformer = feature_transformer
        for a in env.actions_available:
            self.models[a] = SGDRegressor(loss='squared_epsilon_insensitive', penalty='l2',
                                          alpha=0.01, fit_intercept=True,
                                          shuffle=False, epsilon=0.1, learning_rate='optimal',
                                          eta0=0.01, power_t=0.25)
        self.bloom_states = BloomFilter(max_elements=256**2)

    def predict_from_action(self, s, a):
        X = self.feature_transformer.transform(s, normalize=True)
        try:
            y = self.models[a].predict(X)[0]
        except:
            y = 0.0

        return y

    def predict(self, s):
        return np.array([self.predict_from_action(s, a) for a in self.models.keys()])

    def update(self, s, a, G, gamma=0.99, lambda_=0.7):
        X = self.feature_transformer.transform(s, normalize=True)
        self.models[a].partial_fit(np.array([X]), np.array([G]))

    def sample_action(self, s, eps):
        x = tuple(self.feature_transformer.transform(s, normalize=False))
        if x in self.bloom_states:
            # Maybe it's a seen state
            actions = self.models.keys()
            G = [self.predict_from_action(s, a) for a in self.models.keys()]
            if np.max(G) > 1.0:
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


class QubeBloomDualRegAgent(object):
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = {}
        self.models_elite = {}
        self.feature_transformer = feature_transformer
        for a in env.actions_available:
            self.models[a] = SGDRegressor(loss='epsilon_insensitive', penalty='l2',
                                          alpha=0.001, fit_intercept=True,
                                          shuffle=False, epsilon=0.01, learning_rate='constant',
                                          eta0=0.01, power_t=0.25)
            self.models_elite[a] = SGDRegressor(loss='epsilon_insensitive', penalty='l2',
                                                alpha=0.001, fit_intercept=True,
                                                shuffle=False, epsilon=0.01, learning_rate='constant',
                                                eta0=0.01)
        self.bloom_states = BloomFilter(max_elements=256**2)

    def predict(self, s):
        return np.array([self.predict_from_action(s, a) for a in self.models.keys()])

    def predict_from_action(self, s, a):
        X = self.feature_transformer.transform(s, normalize=True)
        try:
            y = self.models[a].predict(X)[0]
        except:
            y = 0.0

        return y

    def predict_from_action_elite(self, s, a):
        X = self.feature_transformer.transform(s, normalize=True)
        try:
            y = self.models_elite[a].predict(X)[0]
        except:
            y = 0.0

        return y

    def update(self, s, a, G, gamma=0.99, lambda_=0.7):
        X = self.feature_transformer.transform(s, normalize=True)
        self.models[a].partial_fit(np.array([X]), np.array([G]))

        if G > 1.0:
            self.models_elite[a].partial_fit(np.array([X]), np.array([G]))

    def sample_action(self, s, eps):
        x = tuple(self.feature_transformer.transform(s, normalize=False))
        if x in self.bloom_states:
            # Maybe it's a seen state
            actions = self.models.keys()
            G = [self.predict_from_action(s, a) for a in self.models.keys()]
            max_G = np.max(G)
            G_elite = [self.predict_from_action_elite(s, a) for a in self.models_elite.keys()]
            max_G_elite = np.max(G_elite)
            if (max_G_elite - max(0.0, max_G)) > 0.5 and max_G_elite >= 1.0:
                print "Taking an elitist action!"
                a = actions[np.argmax(G_elite)]
                return a
            else:
                # First, I need to update the elite models
                a = actions[np.argmax(G)]
                #X = self.feature_transformer.transform(s, normalize=True)
                #self.models_elite[a].partial_fit(np.array([X]), np.array([max_G]))

                if max_G > 1.0:
                    # It's a good state to exploit
                    #self.bloom_states.add(x)
                    return a
                else:
                    # It's not important, so we can explore
                    #self.bloom_states.add(x)
                    return self.env.sample_action()
        else:
            # It's not a seen state, so we can explore
            self.bloom_states.add(x)
            return self.env.sample_action()


class QubeBloomPARAgent(object):
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = {}
        self.feature_transformer = feature_transformer
        for a in env.actions_available:
            self.models[a] = PassiveAggressiveRegressor(C=1.0, fit_intercept=True, shuffle=False)
        self.bloom_states = BloomFilter(max_elements=256**2)
        self.nonseen_states = 0

    def predict(self, s):
        return np.array([self.predict_from_action(s, a) for a in self.models.keys()])

    def predict_from_action(self, s, a):
        X = self.feature_transformer.transform(s, normalize=True)
        try:
            y = self.models[a].predict(X)[0]
        except:
            y = 0.0

        return y

    def update(self, s, a, G, gamma=0.99, lambda_=0.7):
        X = self.feature_transformer.transform(s, normalize=True)
        self.models[a].partial_fit(np.array([X]), np.array([G]))

    def sample_action(self, s, eps):
        x = tuple(self.feature_transformer.transform(s, normalize=False))
        if x in self.bloom_states:
            # Maybe it's a seen state
            actions = self.models.keys()
            G = [self.predict_from_action(s, a) for a in self.models.keys()]
            if np.max(G) > 1.0:
                # It's a good state to exploit
                self.bloom_states.add(x)
                return actions[np.argmax(G)]
            else:
                # It's not important, so we can explore
                self.bloom_states.add(x)
                return self.env.sample_action()
        else:
            # It's not a seen state, so we can explore
            self.nonseen_states += 1
            self.bloom_states.add(x)
            return self.env.sample_action()


class QubeBloomDualPARAgent(object):
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = {}
        self.models_elite = {}
        self.feature_transformer = feature_transformer
        for a in env.actions_available:
            self.models[a] = PassiveAggressiveRegressor(C=1.0, fit_intercept=True, shuffle=False,
                                                        loss='epsilon_insensitive', epsilon=0.1)
            self.models_elite[a] = PassiveAggressiveRegressor(C=1.0, fit_intercept=True, shuffle=False,
                                                              loss='epsilon_insensitive', epsilon=0.1)
        self.bloom_states = BloomFilter(max_elements=256**2)

    def predict(self, s):
        return np.array([self.predict_from_action(s, a) for a in self.models.keys()])

    def predict_from_action(self, s, a):
        X = self.feature_transformer.transform(s, normalize=True)
        try:
            y = self.models[a].predict(X)[0]
        except:
            y = 0.0

        return y

    def predict_from_action_elite(self, s, a):
        X = self.feature_transformer.transform(s, normalize=True)
        try:
            y = self.models_elite[a].predict(X)[0]
        except:
            y = 0.0

        return y

    def update(self, s, a, G, gamma=0.99, lambda_=0.7):
        X = self.feature_transformer.transform(s, normalize=True)
        self.models[a].partial_fit(np.array([X]), np.array([G]))

        if G > 1.0:
            self.models_elite[a].partial_fit(np.array([X]), np.array([G]))

    def sample_action(self, s, eps):
        x = tuple(self.feature_transformer.transform(s, normalize=False))
        if x in self.bloom_states:
            # Maybe it's a seen state
            actions = self.models.keys()
            G = [self.predict_from_action(s, a) for a in self.models.keys()]
            max_G = np.max(G)
            G_elite = [self.predict_from_action_elite(s, a) for a in self.models_elite.keys()]
            max_G_elite = np.max(G_elite)
            if (max_G_elite - max(0.0, max_G)) > 0.5 and max_G_elite >= 1.0:
                #print "Taking an elitist action!"
                a = actions[np.argmax(G_elite)]
                return a
            else:
                # First, I need to update the elite models
                a = actions[np.argmax(G)]
                #X = self.feature_transformer.transform(s, normalize=True)
                #self.models_elite[a].partial_fit(np.array([X]), np.array([max_G]))

                if max_G > 0.0:
                    # It's a good state to exploit
                    #self.bloom_states.add(x)
                    return a
                else:
                    # It's not important, so we can explore
                    #self.bloom_states.add(x)
                    return self.env.sample_action()
        else:
            # It's not a seen state, so we can explore
            self.bloom_states.add(x)
            return self.env.sample_action()


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