#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

__author__ = 'leferrad'

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Lambda
from keras.optimizers import RMSprop, SGD, Adam
from keras.regularizers import l2, l1
import random


class ExperienceReplay(object):
    def __init__(self, max_items=500):
        self.buffer = {}
        self.max_items = max_items

    def add_state(self, state, value):
        # Overwrite existing value if the state is already stored
        self.buffer[state] = value

        if len(self.buffer) > self.max_items:
            min_item = min(self.buffer.items(), key=lambda i: i[1])
            self.buffer.pop(min_item[0])

    def get_sample(self, max_items=50):
        n_items = min(len(self.buffer), max_items)
        return random.sample(self.buffer.items(), n_items)


class MLPRegressor(object):
    def __init__(self, D, lr=1e-4):
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', kernel_initializer='normal', kernel_regularizer=l1(0.001),
                             input_shape=(D,)))
        self.model.add(Dense(8, activation='linear', kernel_initializer='normal', kernel_regularizer=l2(0.001)))
        self.model.add(Dense(1, activation='linear', kernel_initializer='normal'))
        # For a mean squared error regression problem
        #self.optimizer = RMSprop(lr=lr)
        #self.optimizer = SGD(lr=lr, nesterov=True)
        self.optimizer = Adam(lr=lr)
        self.loss = 'mean_squared_error'
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs', write_graph=True)

    def partial_fit(self, x, y):
        if isinstance(x, np.ndarray) is False:
            x = np.array(x)
        if isinstance(y, np.ndarray) is False:
            y = np.array(y)
        self.model.fit(x=np.array([x]), y=np.array([y]), epochs=1, batch_size=1, verbose=0)

    def predict(self, x):
        return self.model.predict(np.array([x]))[0][0]


class MLPRegressorExperience(MLPRegressor):
    def __init__(self, D, lr=1e-4, max_memory=1000):
        MLPRegressor.__init__(self, D, lr)
        self.memory = ExperienceReplay(max_memory)
        self.steps = 0

    def partial_fit(self, x, y):
        if x[0] < 1.0:
            # De-normalize
            x = [int(x_i * 255) for x_i in x]
        x = tuple(x)
        self.memory.add_state(x, y)

        if self.steps > 50:
            # After 100 steps, it's time to fit the model
            train_data = self.memory.get_sample(max_items=30)
            x, y = [], []

            for x_i, y_i in train_data:
                x.append([i / 255.0 for i in x_i])
                y.append(y_i)


            self.model.fit(x=np.array(x), y=np.array(y), epochs=5, batch_size=6, verbose=0)
            self.steps = 1
        else:
            self.steps += 1


class QubeRegAgent(object):
    def __init__(self, env, feature_transformer):
        # TODO: store lambda value here!
        self.env = env
        self.models = {}
        self.feature_transformer = feature_transformer
        for a in env.actions_available:
            self.models[a] = MLPRegressorExperience(feature_transformer.dimensions)
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
        self.models[a].partial_fit(X, [G])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.sample_action()
        else:
            actions = self.models.keys()
            G = [self.predict_from_action(s, a) for a in self.models.keys()]
            return actions[np.argmax(G)]


class QubeBoltzmannRegAgent(QubeRegAgent):
    def __init__(self, env, feature_transformer, t=10):
        QubeRegAgent.__init__(self, env, feature_transformer)
        self.t = t

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
        a = np.random.choice(actions, p=probability_actions)
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