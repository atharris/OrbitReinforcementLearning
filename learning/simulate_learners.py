from DQN import DQN
from SARSA import SARSA
import gym
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
import os
from PIL import Image, ImageFont, ImageDraw
import time

if __name__ == '__main__':
	# Load rewards lists
	# rList_DDQN = np.load('tests/CartPole-v0_DDQN_003/QCartPole_DDQN_Training_Reward_List.npy')
	rList_DQN = np.load('tests/CartPole-v0_DQN_004/QCartPole_DQN_Training_Reward_List.npy')
	# rList_SARSA = np.load('tests/CartPole-v0_SARSA_008/QCartPole_SARSA_Training_Reward_List.npy')

	env = gym.make('CartPole-v0')
	a_size = env.action_space.n
	s_size = 4
	episodes = [1, 50, 100, 150, 175, 200, 250, len(rList_DQN)-1]
	steps = 201
	rList = np.zeros(len(episodes))
	env.reset()
	QL = SARSA(s_size, a_size, hneurons=[24, 24], replay_buffer=10000, loss='DDQN')
	pause=0.01
	fnt = ImageFont.truetype('gifs/open-sans/OpenSans-Regular.ttf', 24)
	gif_path = 'gifs/CartPole-v0_DQN_004/'
	render = True
	save_rewards = False

	for episode in tqdm(episodes, desc="simulating"):
		s = env.reset()
		filename_NN = 'tests/CartPole-v0_DQN_004/episode_' + '{0:0>5}'.format(episode) + '_model_reward_' + str(rList_SARSA[episode])
		filename_gif = gif_path + 'episode_' + '{0:0>5}'.format(episode) + '.gif'
		QL.load(filename_NN)
		rsum = 0
		s = np.array(s)
		a = QL.action(s, 0)
		with imageio.get_writer(filename_gif, mode='I') as writer:
			for step in range(steps):
				if render:
					# Capture the render and save it to a .gif
					capture = env.render(mode='rgb_array')
					im = Image.fromarray(capture)
					# Add episode number to the image
					label = 'Episode ' + str(episode) + ', Reward = {}'.format(rsum)
					d = ImageDraw.Draw(im)
					d.text((10,10), label, font=fnt, fill=(105,105,105))
					im.save(gif_path+'gif_frame.jpeg')
					image = imageio.imread(gif_path+'gif_frame.jpeg')
					writer.append_data(image)
					os.remove(gif_path+'gif_frame.jpeg')

				# Propogate simulation
				s1, r, done, _ = env.step(a)
				rsum += r
				s1 = np.array(s1)
				a1 = QL.action(s1, 0)
				s = s1
				a = a1
				# time.sleep(pause)
				if done:
					# rList[episode] = rsum
					break
	if save_rewards:
		np.save('tests/CartPole-v0_DQN_004/QCartPole_SARSA_Simulation_Reward_List.npy', rList)