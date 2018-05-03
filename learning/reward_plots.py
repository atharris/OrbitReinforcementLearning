from DQN import DQN
from SARSA import SARSA
import gym
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
import os

if __name__ == '__main__':
	
	# load the training reward history data
	path = 'gifs'
	rList_DDQN = np.load('tests/CartPole-v0_DDQN_003/QCartPole_DDQN_Training_Reward_List.npy')
	rList_DQN = np.load('tests/CartPole-v0_DQN_004/QCartPole_DQN_Training_Reward_List.npy')
	rList_SARSA = np.load('tests/CartPole-v0_SARSA_008/QCartPole_SARSA_Training_Reward_List.npy')

	# plot the data one episode at a time and add them to a gif
	episodes = np.max([len(rList_DDQN), len(rList_DQN), len(rList_SARSA)])
	filename = path+'/rewards.png'
	plt.figure()
	plt.xlabel('episodes')
	plt.ylabel('rewards')
	line_DQN, = plt.plot(rList_DQN, label='DQN')
	line_DDQN, = plt.plot(rList_DDQN, label='DDQN')
	line_SARSA, = plt.plot(rList_SARSA, label='SARSA')
	plt.legend(handles=[line_DQN, line_DDQN, line_SARSA])
	plt.xlim([0, episodes])
	plt.ylim([0, 205])
	plt.savefig(filename)
	plt.close()
	# with imageio.get_writer('gifs/training_rewards.gif', mode='I') as writer:
		# for i in tqdm(range(0, episodes, 3), desc='plotting'):
		# 	plt.figure()
		# 	plt.xlabel('episodes')
		# 	plt.ylabel('rewards')
		# 	line_DQN, = plt.plot(rList_DQN[0:i], label='DQN')
		# 	line_DDQN, = plt.plot(rList_DDQN[0:np.min([i, len(rList_DDQN)])], label='DDQN')
		# 	line_SARSA, = plt.plot(rList_SARSA[0:np.min([i, len(rList_SARSA)])], label='SARSA')
		# 	plt.legend(handles=[line_DQN, line_DDQN, line_SARSA])
		# 	plt.xlim([0, episodes])
		# 	plt.ylim([0, 205])
		# 	plt.savefig(filename)
		# 	plt.close()
		# 	image = imageio.imread(filename)
		# 	writer.append_data(image)
		# 	os.remove(filename)
		# extra_frames = 1
		# for i in tqdm(range(extra_frames), desc='adding extra frames'):
			# plt.figure()
			# plt.xlabel('episodes')
			# plt.ylabel('rewards')
			# line_DQN, = plt.plot(rList_DQN, label='DQN')
			# line_DDQN, = plt.plot(rList_DDQN, label='DDQN')
			# line_SARSA, = plt.plot(rList_SARSA, label='SARSA')
			# plt.legend(handles=[line_DQN, line_DDQN, line_SARSA])
			# plt.xlim([0, episodes])
			# plt.ylim([0, 205])
			# plt.savefig(filename)
			# plt.close()
			# image = imageio.imread(filename)
			# writer.append_data(image)

	# env = gym.make('CartPole-v0')
	# a_size = env.action_space.n
	# s_size = 4
	# # N_SARSA = 9
	# # N_DQN = 10
	# N_DDQN = 10
	# env.reset()