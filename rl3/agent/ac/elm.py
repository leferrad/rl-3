#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Policy iteration based on ELM models"""

__author__ = 'leferrad'

# See http://mi.eng.cam.ac.uk/~mg436/LectureSlides/MLSALT7/L5.pdf
# See https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2013-2.pdf


from pyoselm.core import OSELMRegressor, OSELMClassifier

import numpy as np
import random


class SGDRegressor(object):
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


class SGDRegressorDisaster(object):
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


class ExperienceReplay(object):
    def __init__(self, max_items=5000):
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

    def __getattr__(self, item):
        if item in self.buffer:
            return self.buffer[item]
        else:
            return None


class ELMRegressorExperience(object):
    def __init__(self, n_hidden=50, activation_func='tanh', max_memory=5000):
        self.n_hidden = n_hidden
        self.activation_func = activation_func
        self.model = OSELMRegressor(n_hidden=n_hidden, activation_func=activation_func)
        self.model_sec = SGDRegressor(D=6)
        self.memory = ExperienceReplay(max_memory)
        self.fitted = False

    def partial_fit(self, x, y, e=None):
        # TODO: hacer algo con el valor de e!
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

            self.model.fit(X=x, y=np.asarray(y).ravel())
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


class ELMRegressorAC(object):
    def __init__(self, n_hidden=50, activation_func='tanh'):
        self.n_hidden = n_hidden
        self.activation_func = activation_func
        self.model = OSELMRegressor(n_hidden=n_hidden, activation_func=activation_func)
        self.model_sec = SGDRegressor(D=6)
        self.buffer = []
        self.fitted = False

    def partial_fit(self, x, y, e=None):
        # TODO: hacer algo con el valor de e!

        if self.fitted is False:
            # Then we have to wait until a number of 'n_hidden' samples arrives
            self.model_sec.partial_fit(x, y, e)
            self.buffer.append((x, y))

            if len(self.buffer) >= self.n_hidden:
                # Now we can fit the model for first time, and from here start online learning
                x, y = [], []

                for x_i, y_i in self.buffer:
                    x.append(x_i)
                    y.append(y_i)

                self.model.fit(X=x, y=np.asarray(y).ravel())
                self.fitted = True
        else:
            # We can make online learning
            self.model.fit(X=x, y=np.asarray(y).ravel())

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



class ELMRegressorACExperience(object):
    def __init__(self, n_hidden=50, activation_func='tanh', max_memory=5000):
        self.n_hidden = n_hidden
        self.activation_func = activation_func
        self.model = OSELMRegressor(n_hidden=n_hidden, activation_func=activation_func)
        self.model_sec = SGDRegressor(D=6)
        self.memory = ExperienceReplay(max_memory)
        self.fitted = False
        self.steps = 0

    def partial_fit(self, x, y, e=None):
        # TODO: hacer algo con el valor de e!
        self.memory.add_state(tuple(x), y)

        if self.fitted is False or self.steps > self.n_hidden:
            # Then we have to wait until a number of 'n_hidden' samples arrives
            self.model_sec.partial_fit(x, y, e)

            if len(self.memory) >= self.n_hidden:
                # Now we can fit the model for first time, and from here start online learning
                train_data = self.memory.get_sample(max_items=self.n_hidden)
                x, y = [], []

                for x_i, y_i in train_data:
                    x.append(x_i)
                    y.append(y_i)

                self.model.fit(X=x, y=np.asarray(y).ravel())
                self.fitted = True
                self.steps = 0
        else:
            # We can make online learning
            self.model.fit(X=x, y=np.asarray(y).ravel())
            self.steps += 1

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


class PolicyModel(object):
    def __init__(self, env, feature_transformer,
                 t=10, lambda_=0.3, gamma=0.99):
        self.env = env
        self.feature_transformer = feature_transformer
        self.models = {}
        for a in env.actions_available:
            self.models[a] = SGDRegressor(feature_transformer.dimensions)
        self.eligibility = np.zeros((env.n_actions, feature_transformer.dimensions))
        self.t = t
        self.lambda_ = lambda_
        self.gamma = gamma

    def predict_from_action(self, s, a):
        X = self.feature_transformer.transform(s, normalize=True)
        return self.models[a].predict(X)

    def predict(self, s):
        return [self.predict_from_action(s, a) for a in self.models.keys()]

    def get_boltzmann_values(self, s):
        p = self.predict(s)
        max_p = max(p)
        exp_p = [np.exp((p_i - max_p)/float(self.t)) for p_i in p]
        sum_exp_p = sum(exp_p)
        softmax_p = [exp_p_i / float(sum_exp_p) for exp_p_i in exp_p]
        return softmax_p

    def sample_action(self, s):
        probability_actions = self.get_boltzmann_values(s)
        actions = self.env.actions_available.keys()
        return np.random.choice(actions, p=probability_actions)

    def update(self, s, a, V):
        X = self.feature_transformer.transform(s, normalize=True)

        self.eligibility *= self.gamma*self.lambda_
        index_a = self.env.actions_available.keys().index(a)
        self.eligibility[index_a] += X
        self.models[a].partial_fit(X, [V], self.eligibility[index_a])
        #self.models[a].partial_fit(X, [V], 1.0)


class ValueModel(object):
    def __init__(self, env, feature_transformer,
                 n_hidden=50, activation_func='gaussian',
                 t=10, lambda_=0.3, gamma=0.99):
        self.env = env
        self.feature_transformer = feature_transformer
        self.model = ELMRegressorACExperience(n_hidden=n_hidden, activation_func=activation_func)
        self.eligibility = np.zeros((env.n_actions, feature_transformer.dimensions))
        self.t = t
        self.lambda_ = lambda_
        self.gamma = gamma

    def update(self, s, a, V):
        X = self.feature_transformer.transform(s, normalize=True)

        self.eligibility *= self.gamma*self.lambda_
        index_a = self.env.actions_available.keys().index(a)
        self.eligibility[index_a] += X
        self.model.partial_fit(X, [V], self.eligibility[index_a])

    def predict(self, s):
        X = self.feature_transformer.transform(s, normalize=True)
        return self.model.predict(X)


def play_one(policy_estimator, value_estimator, env, max_iters=100):
    env.actions_taken = []  # Reset actions taken on the scramble stage
    observation = env.get_state()

    total_reward = 0
    iters = 0

    while not env.is_solved() and iters < max_iters:
        # Take a step
        action = policy_estimator.sample_action(observation)
        next_observation, reward, solved = env.take_action(action)
        total_reward += reward

        # Calculate TD Target
        next_value = value_estimator.predict(next_observation)
        td_target = reward + value_estimator.gamma * next_value
        td_error = td_target - value_estimator.predict(observation)

        # Update the value estimator
        value_estimator.update(observation, action, td_target)

        # Update the policy estimator
        # using the td error as our advantage estimate
        policy_estimator.update(observation, action, td_error)

        if env.is_solved():
            print "WOW! The cube is solved! Algorithm followed: %s" % str(env.actions_taken)

        iters += 1

    return total_reward, iters