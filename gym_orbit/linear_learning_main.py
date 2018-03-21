import gym
import gym_orbit
import numpy as np


env = gym.make('linear_orbit-v0')

#   Test action space
act_space = env.action_space.n

stateList = []
rewardList = []
dList = []

#   Test each action
for action in act_space:
    tmpState, tmpRew, tempD = env.step(action)
    stateList.append(tmpState)
    rewardList.append(tmpRew)
    dList.append(tempD)

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .8
y = .95
num_episodes = 2000
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        #Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Get new state and reward from environment
        s1,r,d,_ = env.step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            break
    #jList.append(j)
    rList.append(rAll)

print "Score over time: " +  str(sum(rList)/num_episodes)

print "Final Q-Table Values"
print Q
