from DQN import DQN
from SARSA import SARSA
import gym
import datetime
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	a_size = env.action_space.n
	s_size = 4
	N_SARSA = 3
	N_DQN = 1
	now = datetime.datetime.now()
	env.reset()

	for ii in range(N_SARSA):
		path = "tests/CartPole-v0_SARSA_" + "{0:0>3}".format(ii)
		print("Calling SARSA...")
		QL = SARSA(s_size, a_size, hneurons=[24, 24], replay_buffer=10000)
		print("SARSA learner initialized")
		rList = QL.train(env, episodes=3000, steps=201, target_update_steps=100, y=0.99, random_eps_frac=0.25, 
						 model_logging=True, folder_name=path, reward_threshold=0)
		print("done training.")
		QL.save(path+'/QCartPole_SARSA_final')
		np.save(path+'/QCartPole_SARSA_Training_Reward_List.npy', rList)
		plt.figure()
		plt.xlabel('episodes')
		plt.ylabel('rewards')
		plt.plot(rList)
		plt.savefig(path+'QCartPole_SARSA_final.png')
		env.reset()

	# Test the trained QLearner
	# runs = 10
	# rewards, num_steps = QL.simulate(env, epochs=runs, steps=100, pause=0.5, render=True)
	# rewards, num_steps, _, __  = QL.simulate(env, episodes=runs, steps=100, pause=0.02, render=True)