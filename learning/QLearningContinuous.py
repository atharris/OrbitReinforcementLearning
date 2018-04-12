import tensorflow as tf
import numpy as np
import random
import gym
import time
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from gym.envs.registration import register

class QLearnContinuous(object):

	def __init__(self, s_size, a_size, hneurons=[10,10]):
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
		return

	def make_model(self, hneurons, dropout=0, activ_fun='relu'):
		# Initialize model and add the input and first hidden layer
		self.model = Sequential()
		self.model.add(Dense(hneurons[0], init='lecun_uniform', input_shape=(self.s_size,)))
		self.model.add(Activation(activ_fun))
		#self.model.add(Dropout(0.2)) # Use dropout?

		# Add each additional layer to the net
		for ii in range(len(hneurons)-1):
			self.model.add(Dense(hneurons[ii+1], init='lecun_uniform'))
			self.model.add(Activation(activ_fun))
			#self.model.add(Dropout(0.2)) # Use dropout?

		# Add in the output layer
		self.model.add(Dense(self.a_size, init='lecun_uniform'))
		self.model.add(Activation('linear'))

		# Reset all the model weights
		rms = RMSprop()
		self.model.compile(loss='mse', optimizer=rms)

		return self.model

	def action(self, s, epsilon):
		"""
		Select the best action given the state and odds of a random action.
		"""
		if (random.random() < epsilon): # choose a random action
			action = np.random.randint(0,self.a_size)
		else: # pick the largest Q value predicted by the model
			s = np.array(s)
			Qval = self.model.predict(s.reshape(1,self.s_size), batch_size=1)
			action = np.argmax(Qval)

		return action

	def _remember(self, memory):
		if (len(self.replay) < self.buffer):
			self.replay.append(memory)
		else:
			if (self.ii_mem < (buffer-1)):
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
			s = np.array(s)
			s1 = np.array(s1)
			oldQ = self.model.predict(s.reshape(1,self.s_size), batch_size=1)
			newQ = self.model.predict(s1.reshape(1,self.s_size), batch_size=1)
			maxQ = np.max(newQ)
			y = np.zeros((1,self.a_size))
			y[:] = oldQ[:]
			if done:
				update = r
			else:
				update = r + self.y*maxQ
			y[0][a] = update
			# print('y = {}'.format(y))
			X_train.append(s.reshape(self.s_size,))
			y_train.append(y.reshape(self.a_size,))


		return X_train, y_train

	def train(self, env, buffer=40, batch_size=20,epochs=1000, steps=100, eps_range=[1.0, 0.1], y=0.9):
		"""
		Q Learn in the input environment for a set number of epochs and steps.
		Input parameters include:
		
		"""
		epsilon = eps_range[0]
		self.buffer = buffer
		self.y = y
		for epoch in range(epochs):
			# Initial state observation on reset 
			s = env.reset()
			for step in range(steps):
				# Determine optimal action
				a = self.action(s, epsilon)
				# Take action and observe the result
				s1, r, done, _ = env.step(a)
				# Remember the experience
				self._remember([s, a, r, s1, done])

				# Train the model if there are enough memories
				if(len(self.replay) == self.buffer):
					minibatch = random.sample(self.replay, batch_size)
					X_train, y_train = self.trainingData(minibatch)
					# print('X_train has dimensions: {}'.format(np.shape(X_train)))
					# print('y_train has dimensions: {}'.format(np.shape(y_train)))
					self.model.fit(X_train, y_train, batch_size=1, epochs=1, verbose=1)
				s = s1
				# Stop if the environment is done with this epoch
				if done:
					break
			if epsilon > eps_range[1]:
				epsilon -= (1/epochs)

		return self

	def simulate(self, env, epochs=10, steps=100, render=True, pause=1):
		"""
		Simulate an agent in the environment using the current Q model.
		"""
		r_hist = np.zeros(epochs)
		step_totals = []
		for epoch in range(epochs):
			s = env.reset()
			rsum = 0
			for step in range(steps):
				if render:
					env.render()
				a = self.action(s, 0)
				s1, r, done, _ = env.step(a)
				rsum += r
				s = s1
				time.sleep(pause)
				if done:
					r_hist.append(rsum)
					step_total.append(step)
					break


		return r_hist, step_total

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

	grid_size = 4
	# env = gym.make("FrozenLake8x8NotSlippery-v0")
	env = gym.make("FrozenLakeNotSlippery-v0")
	a_size = env.action_space.n
	s_size = 1
	print("Calling QLearnContinuous...")
	QL = QLearnContinuous(s_size, a_size, [164, 150])
	print("Qlearner initialized.")
	rList = QL.train(env, epochs=2000, steps=100)
	print("done training.")
	print()
	env.reset()
	env.render()
	print()

	for i in range(grid_size):
		row = list()
		for j in range(grid_size):
			a = QL.action(grid_size*i + j, 0)
			direct = print_direction(a)
			row.append(direct)
		print('{}'.format(row))

	# Test the trained QLearner
	runs = 100
	rewards, num_steps = QL.simulate(env, epochs=runs, steps=100)

	print()
	print("In {} test runs, the learner found the goal {} percent of the time.".format(runs, np.sum(rewards)))
	print("The average number of steps was {}".format(np.mean(num_steps)))