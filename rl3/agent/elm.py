#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

__author__ = 'leferrad'

from pyoselm.core import OSELMRegressor

import numpy as np
import random


class ExperienceReplay(object):
    def __init__(self, max_items=500):
        self.buffer = {}
        self.max_items = max_items

    def add_state(self, state, value):
        # Overwrite existing value if the state is already stored
        self.buffer[state] = value

        if len(self.buffer) > self.max_items:
            # Forget the min value
            pop_item = min(self.buffer.items(), key=lambda i: i[1])
            # Forget a random value
            #pop_item = random.choice(self.buffer.items())
            self.buffer.pop(pop_item[0])

    def get_sample(self, max_items=50):
        n_items = min(len(self.buffer), max_items)
        return random.sample(self.buffer.items(), n_items)

    def __len__(self):
        return len(self.buffer)


class ExperienceReplayElegibility(ExperienceReplay):
    def __init__(self, max_items=5000):
        ExperienceReplay.__init__(self, max_items)

    def add_state_e(self, state, value, e):
        # Overwrite existing value if the state is already stored
        self.buffer[state] = (value, e)

        if len(self.buffer) > self.max_items:
            # Forget the min value
            pop_item = min(self.buffer.items(), key=lambda i: i[1][0])
            # Forget a random value
            #pop_item = random.choice(self.buffer.items())
            self.buffer.pop(pop_item[0])


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


class SGDRegressorDisaster:
    # Based on http://ftp.bstu.by/ai/To-dom/My_research/Papers-2.1-done/RL/0/FinalReport.pdf
    def __init__(self, D, lr=1e-4, lr_disaster=1e-2, y_disaster=20):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = lr
        self.lr_disaster = lr_disaster
        self.y_disaster = y_disaster

    def partial_fit(self, x, y, e):
        if isinstance(x, np.ndarray) is False:
            x = np.array(x)
        if isinstance(y, np.ndarray) is False:
            y = np.array(y)

        if y[0] >= self.y_disaster:
            lr = self.lr_disaster
        else:
            lr = self.lr

        self.w += lr*(y - x.dot(self.w))*e

    def predict(self, x):
        x = np.array(x)
        return x.dot(self.w)


class SGDRegressorDisasterExperience:
    # Based on http://ftp.bstu.by/ai/To-dom/My_research/Papers-2.1-done/RL/0/FinalReport.pdf
    def __init__(self, D, lr=1e-4, lr_disaster=1e-2, y_disaster=20, max_memory=1000):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = lr
        self.lr_disaster = lr_disaster
        self.y_disaster = y_disaster
        self.memory = ExperienceReplayElegibility(max_memory)
        self.steps = 0

    def _partial_fit_single(self, x, y, e):
        if isinstance(x, np.ndarray) is False:
            x = np.array(x)
        if isinstance(y, np.ndarray) is False:
            y = np.array(y)

        if y[0] >= self.y_disaster:
            lr = self.lr_disaster
        else:
            lr = self.lr

        self.w += lr*(y - x.dot(self.w))*e

    def partial_fit(self, x, y, e):
        if x[0] <= 1.0:
            # De-normalize
            x = [int(x_i * 255) for x_i in x]
        x = tuple(x)
        self.memory.add_state_e(x, y, e)

        if self.steps > 300:
            # After a max of steps, it's time to fit the model
            train_data = self.memory.get_sample(max_items=20)

            for x_i, (y_i, e_i) in train_data:
                # Normalize
                x_i = [i / 255.0 for i in x_i]
                self._partial_fit_single(x_i, y_i, e_i)

            self.steps = 0
        else:
            # Normalize
            x = [i / 255.0 for i in x]
            self._partial_fit_single(x, y, e)

            self.steps += 1

    def predict(self, x):
        x = np.array(x)
        return x.dot(self.w)


class ELMRegressorExperience(object):
    def __init__(self, n_hidden=50, max_memory=5000):
        self.n_hidden = n_hidden
        self.model = OSELMRegressor(n_hidden=n_hidden)
        self.model_sec = SGDRegressor(D=6)
        self.memory = ExperienceReplay(max_memory)
        self.fitted = False

    def partial_fit(self, x, y, e):
        self.model_sec.partial_fit(x, y, e)
        if x[0] < 1.0:
            # De-normalize
            x = [int(x_i * 255) for x_i in x]
        x = tuple(x)
        self.memory.add_state(x, y)

        if len(self.memory) >= self.n_hidden:
            train_data = self.memory.get_sample(max_items=self.n_hidden)
            x, y = [], []

            for x_i, y_i in train_data:
                # Normalize
                x.append([i / 255.0 for i in x_i])
                y.append(y_i)

            self.model.fit(X=x, y=np.asarray(y).ravel(), eleg=e)
            self.fitted = True

    def predict(self, x):
        if self.fitted:
            y = self.model.predict(x)
            if isinstance(y, np.ndarray):
                y = y[0]
        else:
            y = self.model_sec.predict(x)
            if isinstance(y, np.ndarray):
                y = y[0]
        return y


class QubeRegAgent(object):
    def __init__(self, env, feature_transformer):
        # TODO: store lambda value here!
        self.env = env
        self.models = {}
        self.feature_transformer = feature_transformer
        for a in env.actions_available:
            self.models[a] = SGDRegressorDisaster(feature_transformer.dimensions)
        self.eligibilities = np.zeros((env.n_actions, feature_transformer.dimensions))

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


class QubeBoltzmannRegAgent(object):
    def __init__(self, env, feature_transformer, t=10):
        # TODO: store lambda value here!
        self.env = env
        self.models = {}
        self.feature_transformer = feature_transformer
        for a in env.actions_available:
            self.models[a] = ELMRegressorExperience(n_hidden=50)
            #self.models[a] = RFRegressorExperience()
        self.eligibilities = np.zeros((env.n_actions, feature_transformer.dimensions))
        self.t = t

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

    def get_boltzmann_values(self, s):
        q_values = [self.predict_from_action(s, a) for a in self.models.keys()]
        max_q_value = max(q_values)
        exp_q_values = [np.exp((q - max_q_value)/float(self.t)) for q in q_values]
        sum_exp_q_values = sum(exp_q_values)
        softmax_q_values = [exp_q / float(sum_exp_q_values) for exp_q in exp_q_values]
        return softmax_q_values

    def sample_action(self, s, eps):
        probability_actions = self.get_boltzmann_values(s)
        actions = self.models.keys()
        try:
            a = np.random.choice(actions, p=probability_actions)
        except:
            print("UPS!")
        return a


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