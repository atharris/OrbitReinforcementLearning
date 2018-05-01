import tensorflow as tf
import numpy as np
import random
import gym
import time
import matplotlib.pyplot as plt
import os
from tqdm import tqdm, trange
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop, Adam
from keras.utils import to_categorical
from gym.envs.registration import register

class SARSA(object):

	def __init__(self, s_size, a_size, hneurons=[10,10], onehot=False, lr=0.00025, replay_buffer=10000):
		"""
		Initialize state and action space lengths.  State size is the size of the
		neural net's input layer and the action size is the size of the net ouput
		layer.
		"""
		self.s_size = s_size
		self.a_size = a_size
		self.replay = []
		self.learning_rate = lr
		self.buffer = replay_buffer
		# Create the initial value function (Q) model
		self._make_model(hneurons)
		self.ii_mem = 0 # for the _remember function
		self.onehot = onehot
		return

	def _make_model(self, hneurons, dropout=0, activ_fun='relu'):
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
		self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		self.model.summary()

		# Initialize target model to the same structure as the trained model
		self.target_model = Sequential()
		self.target_model.add(Dense(hneurons[0], activation=activ_fun, input_dim=self.s_size))
		for ii in range(len(hneurons)-1):
			self.target_model.add(Dense(hneurons[ii+1], activation=activ_fun))
		self.target_model.add(Dense(self.a_size, activation='linear'))
		self.target_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		self.target_model.summary()
		return self.model

	def _copy_model_to_target(self):
		self.target_model.set_weights(self.model.get_weights())

	def action(self, s, epsilon):
		"""
		Select the best action given the state and epsilon, the odds of a random action.
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

	def _minibatch(self, batch):
		X_train = []
		y_train = []
		for memory in batch:
			s, a, r, s1, a1, done = memory
			# print("s={}, a={}, r={}, s1={}, a1={}".format(s, a, r, s1, a1))
			oldQ = self.model.predict(s.reshape(1,self.s_size))
			newQ = self.target_model.predict(s1.reshape(1,self.s_size))
			# newQ = self.model.predict(s1.reshape(1,self.s_size))
			# print("oldQ={}, newQ={}".format(oldQ, newQ))
			y = np.zeros((1,self.a_size))
			y[:] = oldQ[:]
			if done:
				target = r
			else:
				# SARSA model target update step
				target = r + self.y*newQ[0][a1]
			y[0][a] = target
			X_train.append(s)
			y_train.append(y.reshape(self.a_size,))

		X_train = np.array(X_train)
		y_train = np.array(y_train)
		return X_train, y_train

	def simulate(self, env, episodes=10, steps=100, render=True, pause=0, epsilon=0, remember=False):
		"""
		Simulate an agent in the environment using the current Q model.
		"""
		r_hist = []
		s_hist = []
		a_hist = []
		step_total = []
		for episode in tqdm(range(episodes), desc="simulating"):
			s = env.reset()
			if self.onehot:
				s = to_categorical(s, num_classes=self.s_size)
			rsum = 0
			s = np.array(s)
			a = self.action(s, epsilon)
			for step in range(steps):
				if render:
					env.render()
				a_hist.append(a)
				s_hist.append(s)
				s1, r, done, _ = env.step(a)
				rsum += r
				if self.onehot:
					s1 = to_categorical(s1, num_classes=self.s_size)
				s1 = np.array(s1)
				a1 = self.action(s, epsilon)
				if remember:
					self._remember([s, a, r, s1, a1, done])
				s = s1
				a = a1
				time.sleep(pause)
				if done:
					r_hist.append(rsum)
					step_total.append(step)
					s_hist.append(s1)
					break

		return r_hist, step_total, s_hist, a_hist

	def train(self, env, episodes=1000, steps=100, render=False, epsilon_range=[1.0, 0.1], random_eps_frac=.1, 
			  lin_anneal_frac=.2, minibatch_size=32, y=0.99, target_update_steps=5000, reward_threshold=180):
		"""
		Description of training
		"""

		# Initialize a replay memory of random_eps stored in memory
		self.simulate(env, episodes=int(np.floor(random_eps_frac*episodes)), render=render, 
					  steps=steps, epsilon=1, remember=True)

		# Copy initial target model to have the same weights as the model
		self._copy_model_to_target()

		# Initialize the target_model update counter
		total_steps = 0

		# Set initial epsilon value
		epsilon = epsilon_range[0]

		# Set discount factor
		self.y = y

		# Initialize a list of rewards
		rList = np.zeros(episodes)

		# Initialize reward plot
		plt.ion()
		fig = plt.figure()
		ax = fig.add_subplot(111)
		line,  = ax.plot(0,0,'r-')

		for episode in tqdm(range(episodes), desc="training"):
			# Reset the environment
			s = env.reset()
			if self.onehot:
				s = to_categorical(s, num_classes=self.s_size)
			s = np.array(s)
			a = self.action(s, epsilon)

			# Check for convergence
			if episode > 25
				avg = np.mean(rList[episode-25:episode])
				if avg > reward_threshold:
					break

			# Run one episode and train the SARSA learner
			for step in range(steps):
				# Take action a
				s1, r, done, _ = env.step(a)
				if self.onehot:
					s1 = to_categorical(s1, num_classes=self.s_size)
				s1 = np.array(s1)

				# Find the next action
				a1 = self.action(s1, epsilon)

				# Store the state, action, reward, state, action (SARSA)
				self._remember([s, a, r, s1, a1, done])

				# Train on a minibatch sampled randomly from the replay memory
				X_train, y_train = self._minibatch(random.sample(self.replay, minibatch_size))
				self.model.fit(X_train, y_train, epochs=1, verbose=0)

				# Every target_update_steps, set target_model equal to model
				total_steps += 1
				if not (total_steps % target_update_steps):
					self._copy_model_to_target()

				# Add up rewards so far
				rList[episode] += r

				# Break if episode is done
				if done:
					# tqdm.write("r={}".format(rList[episode]))
					break

				# Update state and action to the next pair
				s = s1
				a = a1

				# epsilon annealing
				if (epsilon > epsilon_range[1]):
					epsilon -= 1/(lin_anneal_frac*episodes)

			# Plot the most recent reward
			if episode > 2:
				self._update_reward_plot(episode, rList, line, fig, ax)
		return rList[:episode]

	def save(self, filename):
		self.model.save(filename)

	def load(self, filename):
		self.model = load_model(filename)

	def _update_reward_plot(self, episode, rList, line, fig, axes):
		line.set_xdata(range(episode))
		line.set_ydata(rList[:episode])
		fig.canvas.draw()
		fig.canvas.flush_events()
		axes.set_xlim(0, episode)
		axes.set_ylim(np.min(rList), np.max(rList))