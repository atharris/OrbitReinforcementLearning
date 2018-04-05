import gym
import gym_orbit
import numpy as np
from matplotlib import pyplot as plt

env = gym.make('stationkeep_orbit-v0')

#   Test action space
act_space = env.action_space.n

estState = np.zeros(6,env.max_length)
trueState = np.zeros(6,env.max_length)
refState = np.zeros(6,env.max_length)
rewardList = []
episode_over = False

env.reset()

ind=0
#   Test observation action
while episode_over == False:
    action = np.random.randint(0,3,[1])
    tmpState, tmpRew, episode_over, tmpdict = env.step(action)

    estState[:,ind] = tmpState.state_vec
    trueState[:,ind] = tmpdict.true_state.state_vec
    refState[:,ind] = tmpdict.ref_state.state_vec
    rewardList.append(tmpRew)
    ind=ind+1

fig, axarr = plt.subplots(6, sharex=True)
yArr = ['$x$ (m)', '$y$ (m)', '$z$ (m)', '$\dot{x}$ (m/s)', '$\dot{y}$ (m/s)', '$\dot{z}$ (m/s)']
for plotInd in range(0, 6):
    legendAppend = str(plotInd)
    if plotInd < 6:
        axarr[plotInd].plot(estState[plotInd, :] - trueState[plotInd, :],label='estimator error')
        axarr[plotInd].plot(refState[plotInd, :] - trueState[plotInd, :],label='ctrl error')
        axarr[plotInd].plot(estState[plotInd, :] - refState[plotInd, :], label='sc error')
    axarr[plotInd].set_ylabel(yArr[plotInd])
    axarr[plotInd].grid()
plt.legend()
axarr[-1].set_xlabel('Simulation Timestep')
plt.show()
