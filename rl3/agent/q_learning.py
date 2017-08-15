#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

__author__ = 'leferrad'

from rl3.agent.feature import LBPFeatureTransformer

from sklearn.linear_model import PassiveAggressiveRegressor
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
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 10e-2

    def partial_fit(self, X, Y):
        if isinstance(X, np.ndarray) is False:
            X = np.array(X)
        if isinstance(Y, np.ndarray) is False:
            Y = np.array(Y)
        self.w += self.lr*(Y - X.dot(self.w)) * X

    def predict(self, X):
        if isinstance(X, np.ndarray) is False:
            X = np.array(X)
        return X.dot(self.w)


class QubeRegAgent(object):
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = {}
        self.feature_transformer = feature_transformer
        for a in env.actions_available:
            self.models[a] = SGDRegressor(feature_transformer.dimensions)

    def predict_from_action(self, s, a):
        X = self.feature_transformer.transform(s, normalize=True)
        return self.models[a].predict(X)

    def predict(self, s):
        return np.array([self.predict_from_action(s, a) for a in self.models.keys()])

    def update(self, s, a, G):
        X = self.feature_transformer.transform(s, normalize=True)
        self.models[a].partial_fit(X, [G])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.sample_action()
        else:
            actions = self.models.keys()
            G = [self.predict_from_action(s, a) for a in self.models.keys()]
            return actions[np.argmax(G)]


class QubePARAgent(object):
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = {}
        self.feature_transformer = feature_transformer
        for a in env.actions_available:
            self.models[a] = PassiveAggressiveRegressor(C=1.0, fit_intercept=True, n_iter=10)

    def predict(self, s):
        return np.array([self.predict_from_action(s, a) for a in self.models.keys()])

    def predict_from_action(self, s, a):
        X = self.feature_transformer.transform(s, normalize=True)
        return self.models[a].predict(X)

    def update(self, s, a, G):
        X = self.feature_transformer.transform(s, normalize=True)
        self.models[a].partial_fit(X, [G])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.sample_action()
        else:
            actions = self.models.keys()
            G = [self.predict_from_action(s, a) for a in self.models.keys()]
            return actions[np.argmax(G)]
