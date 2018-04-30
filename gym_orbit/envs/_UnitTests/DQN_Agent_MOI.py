# Deep Q-learning Agent
#   taken directly from: https://keon.io/deep-q-learning/


import keras
# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.optimizers import Adam
from keras.models import Sequential,Input,Model,load_model
from keras.layers import Dense, Dropout, Flatten

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 1.    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.num_eps = 0.0
        self.epsilon_const = 500 #  Number of training runs w/ constant random actions
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                for i in range(len(reward)):
                    target += self.gamma**i*reward[i]
                target += self.gamma**(i+1) * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = load_model(filename)

    def eligibility(self, batch_size, trace_depth):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            j=0
            for state_test, action_test, reward_test, next_state_test, done_test in self.memory:
                if state_test.all() == state.all() and action_test == action and reward_test == reward and next_state_test.all()==next_state.all() and done_test == done:
                    ind = j
                    break
                else:
                    j+=1
            target = reward
            traceBatch= []
            for k in range(trace_depth):
                if ind + k < len(self.memory):
                    traceBatch.append(self.memory[ind + k])
                else:
                    break
            i=0
            if not done:
                for state_t, action_t, reward_t, next_state_t, done_t in traceBatch:
                    if done_t:
                        break
                    target += self.gamma ** i * reward_t
                    i+=1
                target += self.gamma ** (i + 1) * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.num_eps > self.epsilon_const:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        self.num_eps = self.num_eps + 1
