import tensorflow as tf
import numpy as np
import random
import gym
import time
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import Dense, Activation


class QLearn_discrete(object):

	def __init__(self, s_size, a_size, lr=.8, y=.95):
		self.Q = np.zeros((s_size, a_size))
		self.s_size = s_size
		self.a_size = a_size
		self.lr = lr
		self.y = y
		return

	def updateQ(self, s, a, s1, r):
		self.Q[s,a] = self.Q[s,a] + self.lr*(r + self.y*np.max(self.Q[s1,:]) - self.Q[s,a])
		return self.Q

	def action(self, s, stochastic=True, epoch=0):
		if stochastic:
			a = np.argmax(self.Q[s,:] + np.random.randn(1,self.a_size)*(1./(epoch+1)))
		else:
			a = np.argmax(self.Q[s,:])
		return a

	def train(self, env, epochs=10, steps=100):
		rList = []
		rAll = 0
		for i in range(epochs):
			s = env.reset() # initial state observation
			done = False
			for ii in range(steps):
				a = self.action(s, stochastic=True, epoch=i)
				# print("The chosen action is {} for state {}".format(a,s))
				s1, r, done, s2 = env.step(a)
				# penalty for acting without a reward
				# add in penalty for going in a hole
				# if r == 1:
				# 	r = 1 - 0.01*ii
				# if r == 0 and done:
				# 	r = -1

				self.updateQ(s,a,s1,r)
				s = s1
				rAll += r
				if done:
					# print("Epoch {} finished after {} steps.".format(i+1, ii+1))
					break
			rList.append(rAll/(i+1))
		return rList

# Create a class to store sim memory
class ExperienceReplay(object):
	"""
	Object that contains the previous memories of the environment.  Returns training data
	for a ML algorithm or a Q table.
	"""
	def __init__(self, max_memory=1000, state_size=1):
		self.max_memory = max_memory
		self.memory = list()
		self.state_size = state_size
		return

	def remember(self, states, done):
		"""
		states = [state_t1, action_t1, reward_t1, state_t2]
		done = True if the simulation ended here
		"""
		# Save state to memory list
		self.memory.append([states, done])

		# Delete first memory entry if we exceed the max memory size
		if len(self.memory) > self.max_memory:
			del self.memory[0]
		return self

	def sample(self, size=10):
		return np.reshape(np.array(random.sample(self.memory, size)), [size, self.state_size])

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
	env = gym.make("FrozenLake-v0")
	a_size = env.action_space.n
	s_size = env.observation_space.n
	QL = QLearn_discrete(s_size, a_size)
	rList = QL.train(env, epochs=5000)
	print('Q = {}'.format(QL.Q))
	# plt.plot(rList)
	# plt.xlabel('epoch')
	# plt.ylabel('reward per epoch')
	# plt.pause(10)

	env.reset()
	env.render()
	print()
	print()

	for i in range(4):
		row = list()
		for j in range(4):
			a = QL.action(4*i + j, stochastic=False)
			direct = print_direction(a)
			row.append(direct)
		print('{}'.format(row))

	time.sleep(60)
	# Test the trained QLearner
	for i_episode in range(10):
		observation = env.reset()
		for t in range(50):
			os.system('clear')
			env.render()
			action = QL.action(observation, stochastic=False)
			observation, reward, done, info = env.step(action)
			if done:
				print("Episode finished after {} timesteps".format(t+1))
				print("Reward : {}".format(reward))
				time.sleep(3)
				break
			time.sleep(0.1)
			
