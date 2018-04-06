import gym
import time
env = gym.make('FrozenLake-v0')
print(env.action_space)
print(env.observation_space)
for i_episode in range(1):
    observation = env.reset()
    for t in range(15):
        env.render()
        print(observation)
        action = env.action_space.sample()
        print(action)
        observation, reward, done, info = env.step(action)
        # print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break