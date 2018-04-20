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
from keras.optimizers import RMSprop, Adam
from keras.utils import to_categorical
from gym.envs.registration import register
from SARSA import SARSA

class QLearnContinuous(object):

	def __init__(self, s_size, a_size, hneurons=[10,10], onehot=False, lr=0.00025):
		"""
		Initialize state and action space lengths.  State size is the size of the
		neural net's input layer and the action size is the size of the net ouput
		layer.
		"""
		self.s_size = s_size
		self.a_size = a_size
		self.replay = []
		self.learning_rate = lr
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
		rms = RMSprop(lr=self.learning_rate)
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
				target = r
			else:
				target = r + self.y*np.max(newQ)
			y[0][a] = target
			X_train.append(s)
			y_train.append(y.reshape(self.a_size,))

		X_train = np.array(X_train)
		y_train = np.array(y_train)
		return X_train, y_train

	def train(self, env, buffer=80, batch_size=32, epochs=1000, steps=100, eps_range=[1.0, 0.1], decay_range=0.2, y=0.99):
		"""
		Q Learn in the input environment for a set number of epochs and steps.
		Input parameters include:
		
		"""
		rList = np.zeros(epochs)
		epsilon = eps_range[0]
		self.buffer = buffer
		self.y = y
		steps_since_update = 0
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
				steps_since_update += 1
				# Train the model if there are enough memories and a batch_size worth of updates has been made
				# if(len(self.replay) == self.buffer and steps_since_update>=batch_size):
				if(len(self.replay) == self.buffer):
					minibatch = random.sample(self.replay, batch_size)
					X_train, y_train = self.trainingData(minibatch)
					self.model.fit(X_train, y_train, epochs=1, verbose=0)
					steps_since_update = 0
				s = s1
				rList[epoch] += r
				# Stop if the environment is done with this epoch
				if done:
					break
			if epsilon > eps_range[1]:
				epsilon -= 1/(decay_range*epochs)

		return rList

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
	env = gym.make("FrozenLakeNotSlippery-v0")
	# env = gym.make('CartPole-v0')
	# env = gym.make('Blackjack-v0')
	a_size = env.action_space.n
	# s_size = env.observation_space.n
	# s_size = np.shape(env.observation_space)[0]
	s_size = 16
	# print("Calling QLearnContinuous...")
	print("Calling SARSA...")
	# QL = QLearnContinuous(s_size, a_size, [128], onehot=False)
	QL = SARSA(s_size, a_size, [24], onehot=True, replay_buffer=500)
	# print("Qlearner initialized.")
	print("SARSA learner initialized")
	# rList = QL.train(env, epochs=1500, steps=201, buffer=750, batch_size=32, y=0.99)
	# QL.load('QCartPole_3000')
	QL.train(env, episodes=1000, steps=100, target_update_steps=250)
	print("done training.")
	# QL.save('QCartPole_1500')
	print()
	env.reset()
	plt.figure()
	plt.xlabel('epochs')
	plt.ylabel('rewards')
	plt.plot(rList)
	# plt.savefig('QCartPole_1500_rewards.png')
	plt.show()

	# env.render()
	# print()

	for i in range(grid_size):
		row = list()
		for j in range(grid_size):
			s = to_categorical(grid_size*i + j, s_size)
			a = QL.action(s, 0)
			direct = print_direction(a)
			row.append(direct)
		print('{}'.format(row))

	# Test the trained QLearner
	# runs = 1
	# rewards, num_steps = QL.simulate(env, epochs=runs, steps=201, pause=0.05, render=True)
	# rewards = np.array(rewards)
	# num_steps = np.array(num_steps)
	# sum_rewards = np.sum(rewards == 1)
	# sum_draws = np.sum(rewards == 0)
	# sum_losses = np.sum(rewards == -1)
	# print("In {} test runs, the learner won at blackjack {} percent of the time.".format(runs, (sum_rewards/runs)*100))
	# print("{} percent of games were draws.".format(100*sum_draws/runs))
	# print("{} percent of the games were losses".format(100*sum_losses/runs))
	# print("In {} simulation runs, the average number of steps was {}".format(runs,np.mean(num_steps)))