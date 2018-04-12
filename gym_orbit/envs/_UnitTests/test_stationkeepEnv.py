import gym
import gym_orbit
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import DQN_Agent as dqn


env = gym.make('stationkeep_orbit-v0')
episode_over = False
state_size = 6
estState = np.zeros([6,env.max_length+1])
trueState = np.zeros([6,env.max_length+1])
refState = np.zeros([6,env.max_length+1])
rewardList = []
actHist = []
ctrlList = []

obs = env.reset()
state = np.reshape(obs['state'].state_vec, [1, state_size])
colorDict = {0:'blue',
             1:'red'}

ind=-1
#   Test observation action
while episode_over == False:
    action = np.random.randint(0, 2, [1])
    actHist.append(action)
    obs, tmpRew, episode_over, tmpdict = env.step(action)
    next_state = np.reshape(obs['state'].state_vec, [1, state_size])
    state = next_state

    estState[:,ind] = obs['state'].state_vec
    trueState[:,ind] = tmpdict['truth'].state_vec
    refState[:,ind] = tmpdict['ref'].state_vec
    ctrlList.append(obs['control'])
    rewardList.append(tmpRew)
    ind = len(rewardList)

totalCost = [0]
totalReward = [0]
for ind in range(0,env.max_length-1):
    totalCost.append(totalCost[-1] + ctrlList[ind])
    totalReward.append(totalReward[-1]+rewardList[ind])

fig, axarr = plt.subplots(6, sharex=True)
yArr = ['$x$ (km)', '$y$ (km)', '$z$ (km)', '$\dot{x}$ (km/s)', '$\dot{y}$ (km/s)', '$\dot{z}$ (km/s)']
for plotInd in range(0, 6):
    legendAppend = str(plotInd)
    if plotInd < 6:
        axarr[plotInd].plot(abs(estState[plotInd, :] - trueState[plotInd, :]),label='estimator error')
        axarr[plotInd].plot(abs(refState[plotInd, :] - trueState[plotInd, :]),label='ctrl error')
        #axarr[plotInd].plot(estState[plotInd, :] - refState[plotInd, :], label='sc error')
        if plotInd == 0:
            axarr[plotInd].set_title('Estimator, Control Errors vs. Time/Mode')
        for ind in range(0,env.max_length):
            axarr[plotInd].axvspan(ind, ind+1, color=colorDict[int(actHist[ind])], alpha = 0.05)
    axarr[plotInd].set_ylabel(yArr[plotInd])
    axarr[plotInd].grid()
plt.legend()

axarr[-1].set_xlabel('Simulation Step')

plt.figure()
plt.plot(ctrlList,label='$\Delta V$ per mode')
plt.plot(totalCost, label='Total $\Delta V$')
plt.title('$\Delta V$ Usage vs. Time')
plt.grid(True)
plt.ylabel('$\Delta V$ (km/s)')
plt.xlabel('Sim Step')
plt.legend()

plt.figure()
plt.plot(rewardList, label='Mode-wise Reward')
plt.plot(totalReward,label='Summed Reward')
plt.ylabel('Reward')
plt.xlabel('Sim Step')
plt.title('Reward vs. Time')
plt.grid(True)
plt.show()
