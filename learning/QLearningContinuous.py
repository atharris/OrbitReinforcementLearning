import tensorflow as tf
import numpy as np
import random
import gym
import time
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from gym.envs.registration import register

class QLearnContinuous(object):

	def __init__(self, s_size, a_size, hneurons=[10,10], onehot=False):
		"""
		Initialize state and action space lengths.  State size is the size of the
		neural net's input layer and the action size is the size of the net ouput
		layer.
		"""
		self.s_size = s_size
		self.a_size = a_size
		self.replay = []
		self.make_model(hneurons)
		self.ii_mem = 0 # for the _remember function
		self.onehot = onehot
		return

	def make_model(self, hneurons, dropout=0, activ_fun='relu'):
		# Initialize model and add the input and first hidden layer
		self.model = Sequential()
		self.model.add(Dense(hneurons[0], activation=activ_fun, input_dim=self.s_size))
		#self.model.add(Dropout(0.2)) # Use dropout?

		# Add each additional layer to the net
		for ii in range(len(hneurons)-1):
			self.model.add(Dense(hneurons[ii+1], activation=activ_fun))
			#self.model.add(Dropout(0.2)) # Use dropout?

		# Add in the output layer
		self.model.add(Dense(self.a_size, activation='linear'))

		# Reset all the model weights
		rms = RMSprop()
		self.model.compile(loss='mse', optimizer=rms)
		self.model.summary()
		return self.model

	def action(self, s, epsilon):
		"""
		Select the best action given the state and odds of a random action.
		"""
		if (random.random() < epsilon): # choose a random action
			action = np.random.randint(0,self.a_size)
		else: # pick the largest Q value predicted by the model
			Qval = self.model.predict(s.reshape(1,self.s_size))
			action = np.argmax(Qval)

		return action

	def _remember(self, memory):
		if (len(self.replay) < self.buffer):
			self.replay.append(memory)
		else:
			if (self.ii_mem < (self.buffer-1)):
				self.ii_mem += 1
			else:
				self.ii_mem = 0
			self.replay[self.ii_mem] = memory

		return self.replay

	def trainingData(self, batch):
		X_train = []
		y_train = []
		for memory in batch:
			s, a, r, s1, done = memory
			# oldQ = self.model.predict(s.reshape(self.s_size,))
			# newQ = self.model.predict(s1.reshape(self.s_size,))
			# print("s shape = {}".format(np.shape(s)))
			# print("s1 shape = {}".format(np.shape(s1)))
			oldQ = self.model.predict(s.reshape(1,self.s_size))
			newQ = self.model.predict(s1.reshape(1,self.s_size))
			y = np.zeros((1,self.a_size))
			y[:] = oldQ[:]
			if done:
				update = r
			else:
				update = r + self.y*np.max(newQ)
			y[0][a] = update
			X_train.append(s)
			y_train.append(y.reshape(self.a_size,))

		X_train = np.array(X_train)
		y_train = np.array(y_train)
		return X_train, y_train

	def train(self, env, buffer=80, batch_size=40, epochs=1000, steps=100, eps_range=[1.0, 0.02], y=0.9):
		"""
		Q Learn in the input environment for a set number of epochs and steps.
		Input parameters include:
		
		"""
		epsilon = eps_range[0]
		self.buffer = buffer
		self.y = y
		eps_decay = (eps_range[0] - eps_range[1])/eps_range[1]
		for epoch in tqdm(range(epochs), desc="training"):
			# Initial state observation on reset 
			s = env.reset()
			if self.onehot:
				s = to_categorical(s, num_classes=self.s_size)
			s = np.array(s)
			for step in range(steps):
				# Determine optimal action
				a = self.action(s, epsilon)
				# Take action and observe the result
				s1, r, done, _ = env.step(a)
				# Remember the experience
				if self.onehot:
					s1 = to_categorical(s1, num_classes=self.s_size)
				
				s1 = np.array(s1)
				self._remember([s, a, r, s1, done])

				# Train the model if there are enough memories
				if(len(self.replay) == self.buffer):
					minibatch = random.sample(self.replay, batch_size)
					X_train, y_train = self.trainingData(minibatch)
					self.model.fit(X_train, y_train, batch_size=1, epochs=1, verbose=0)
				s = s1
				# Stop if the environment is done with this epoch
				if done:
					break
			if epsilon > eps_range[1]:
				epsilon = eps_range[0]/(1 + eps_decay*epoch/epochs)

		return self

	def simulate(self, env, epochs=10, steps=100, render=True, pause=1):
		"""
		Simulate an agent in the environment using the current Q model.
		"""
		r_hist = []
		step_total = []
		for epoch in tqdm(range(epochs), desc="simulating"):
			s = env.reset()
			if self.onehot:
				s = to_categorical(s, num_classes=self.s_size)
			rsum = 0
			s = np.array(s)
			for step in range(steps):
				if render:
					env.render()
				a = self.action(s, 0)
				s1, r, done, _ = env.step(a)
				rsum += r
				if self.onehot:
					s1 = to_categorical(s1, num_classes=self.s_size)
				s1 = np.array(s1)
				s = s1
				time.sleep(pause)
				if done:
					r_hist.append(rsum)
					step_total.append(step)
					break

		return r_hist, step_total

	def save(self, filename):
		self.model.save(filename)

	def load(self, filename):
		self.model = load_model(filename)

def print_direction(a):
	if a == 0:
		direction = ' Left '
	elif a == 1:
		direction = ' Down '
	elif a == 2:
		direction = ' Right'
	elif a == 3:
		direction = '  Up  '
	return direction

if __name__ == "__main__":
	register(
		id='FrozenLakeNotSlippery-v0',
		entry_point='gym.envs.toy_text:FrozenLakeEnv',
		kwargs={'map_name' : '4x4', 'is_slippery': False},
		max_episode_steps=100,
		reward_threshold=0.78, # optimum = .8196
	)
	register(
		id='FrozenLake8x8NotSlippery-v0',
		entry_point='gym.envs.toy_text:FrozenLakeEnv',
		kwargs={'map_name' : '8x8', 'is_slippery': False},
		max_episode_steps=100,
		reward_threshold=0.78, # optimum = .8196
	)	

	# grid_size = 4
	# env = gym.make("FrozenLake8x8NotSlippery-v0")
	# env = gym.make("FrozenLakeNotSlippery-v0")
	# env = gym.make('CartPole-v0')
	env = gym.make('Blackjack-v0')
	a_size = env.action_space.n
	# s_size = env.observation_space.n
	# s_size = np.shape(env.observation_space)[0]
	s_size = 3
	print("Calling QLearnContinuous...")
	QL = QLearnContinuous(s_size, a_size, [64], onehot=False)
	print("Qlearner initialized.")
	# rList = QL.train(env, epochs=3000, steps=7)
	QL.load('QlearnBlackJack_3000')
	print("done training.")
	# QL.save('QlearnBlackJack_3000')
	print()
	env.reset()
	# env.render()
	# print()

	# for i in range(grid_size):
	# 	row = list()
	# 	for j in range(grid_size):
	# 		s = to_categorical(grid_size*i + j, s_size)
	# 		a = QL.action(s, 0)
	# 		direct = print_direction(a)
	# 		row.append(direct)
	# 	print('{}'.format(row))

	# Test the trained QLearner
	runs = 5000
	rewards, num_steps = QL.simulate(env, epochs=runs, steps=7, pause=0.0, render=False)
	rewards = np.array(rewards)
	num_steps = np.array(num_steps)
	print(rewards[0:25])
	sum_rewards = np.sum(rewards == 1)
	sum_draws = np.sum(rewards == 0)
	sum_losses = np.sum(rewards == -1)
	print("In {} test runs, the learner won at blackjack {} percent of the time.".format(runs, (sum_rewards/runs)*100))
	print("{} percent of games were draws.".format(100*sum_draws/runs))
	print("{} percent of the games were losses".format(100*sum_losses/runs))
	# print("The average number of steps was {}".format(np.mean(num_steps)))