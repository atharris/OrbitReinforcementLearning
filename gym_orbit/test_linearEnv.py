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

action_set = [0,0,1,1,0,0,1,1]
#   Test observation action
for action in action_set:
    tmpState, tmpRew, episode_over, tmpdict = env.step(action)

    obsErr.append(tmpState.est_err)
    stateErr.append(tmpState.state_err)
    errorMode.append(tmpState.error_mode)
    rewardList.append(tmpRew)
    if episode_over:
        print "Something stopped it."
        break



plt.figure()
plt.plot(obsErr,label='estimator error')
plt.plot(stateErr, label='reference error')
plt.plot(errorMode, label='broken sc indicator')
plt.plot(rewardList, label='associated reward')
plt.legend()

plt.show()
