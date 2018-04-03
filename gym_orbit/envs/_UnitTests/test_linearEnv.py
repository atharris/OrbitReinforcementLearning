import gym
import gym_orbit
import numpy as np
from matplotlib import pyplot as plt

env = gym.make('linear_orbit-v0')

#   Test action space
act_space = env.action_space.n

stateList = []
rewardList = []
dList = []
obsErr = []
stateErr = []
errorMode = []

episode_over = False

env.reset()


#   Test observation action
while episode_over == False:
    action = np.random.randint(0,3,[1])
    tmpState, tmpRew, episode_over, tmpdict = env.step(action)

    obsErr.append(tmpState.state_vec[1])
    stateErr.append(tmpState.state_vec[0])
    errorMode.append(tmpState.error_mode)
    rewardList.append(tmpRew)




plt.figure()
plt.plot(obsErr,label='estimator error')
plt.plot(stateErr, label='reference error')
plt.plot(errorMode, label='broken sc indicator')
plt.plot(rewardList, label='associated reward')
plt.legend()

plt.show()
