import gym
import gym_orbit
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import DQN_Agent_MOI as dqn
import os

env = gym.make('mars_orbit_insertion-v0')

#   Test action space
state_size = 21
act_space = 3
episode_over = False
agent = dqn.DQNAgent(state_size, act_space)

rewardplot = []

colorDict = {0:'blue',
             1:'red',
             2:'green'}

ind=-1
num_episodes = 1000
batch_size = 20
depth = 60
#   Begin the training iterations
for ep in range(0,num_episodes):

    obs = env.reset()
    state = np.reshape(np.hstack([[obs['state'].state_vec],[np.diag(obs['state'].covariance)],[obs['ref'].state_vec],[np.array([obs['goal'].a,obs['goal'].e, obs['goal'].omega])]]), [1, state_size])
    # state = np.reshape(np.hstack([[np.array([obs['state'].a,obs['state'].e, obs['state'].omega])],[np.array([obs['ref'].a,obs['ref'].e, obs['ref'].omega])],[np.array([obs['goal'].a,obs['goal'].e, obs['goal'].omega])]]), [1, state_size])
    reward_count = 0
    episode_over = False
    #   Start an episode to keep on lernin
    while episode_over == False:
        action = agent.act(state)
        obs, reward, episode_over, _ = env.step(action)
        next_state = np.reshape(np.hstack([[obs['state'].state_vec],[np.diag(obs['state'].covariance)],[obs['ref'].state_vec],[np.array([obs['goal'].a,obs['goal'].e, obs['goal'].omega])]]), [1, state_size])
        # next_state =  np.reshape(np.hstack([[np.array([obs['state'].a,obs['state'].e, obs['state'].omega])],[np.array([obs['ref'].a,obs['ref'].e, obs['ref'].omega])],[np.array([obs['goal'].a,obs['goal'].e, obs['goal'].omega])]]), [1, state_size])
        agent.remember(state, action, reward, next_state, episode_over)
        reward_count = reward + reward_count
        state = next_state
        if episode_over:
            print("episode:{}/{}, score: {}, e: {:.2}".format(ep, num_episodes, reward_count, agent.epsilon))
            rewardplot.append(reward_count)
    if len(agent.memory) > batch_size:
        agent.eligibility(batch_size, depth)

#   Save the trained model
agent.save('moi_test.h5')

plt.plot(rewardplot)
plt.show()

