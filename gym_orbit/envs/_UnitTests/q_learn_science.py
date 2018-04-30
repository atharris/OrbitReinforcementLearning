import gym
import gym_orbit
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import DQN_Agent as dqn
import os

env = gym.make('stationkeepscience_orbit-v0')

#   Test action space
state_size = 18
act_space = 3
episode_over = False
agent = dqn.DQNAgent(state_size, act_space)

ind=-1
num_episodes = 5000
batch_size = 32

reward_hist = np.zeros([num_episodes,])
#   Begin the training iteration
for ep in range(0,num_episodes):

    obs = env.reset()
    state = np.reshape(np.hstack([[obs['state'].state_vec],[np.diag(obs['state'].covariance)],[obs['ref'].state_vec]]), [1, state_size])
    reward_count = 0
    episode_over = False
    #   Start an episode to keep on lernin
    while episode_over == False:
        action = agent.act(state)
        obs, reward, episode_over, _ = env.step(action)
        next_state = np.reshape(np.hstack([[obs['state'].state_vec],[np.diag(obs['state'].covariance)],[obs['ref'].state_vec]]), [1, state_size])
        agent.remember(state, action, reward, next_state, episode_over)
        reward_count = reward + reward_count

        state = next_state
        if episode_over:
            print("episode:{}/{}, score: {}, e: {:.2}".format(ep, num_episodes, reward_count, agent.epsilon))
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
    reward_hist[ind] = reward_count
    ind = ind+1

#   Save the trained model
agent.save('dqn_zeronoise_science_test.h5')

plt.figure()
plt.plot(reward_hist)
plt.title('Reward vs. Training Episode')
plt.xlabel('Episode')
plt.ylabel('Cumulative reward')
plt.grid(True)
plt.show()

