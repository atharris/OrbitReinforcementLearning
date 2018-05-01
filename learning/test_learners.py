from DQN import DQN
from SARSA import SARSA
import datetime

if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	a_size = env.action_space.n
	s_size = 4
	N_SARSA = 1
	N_DQN = 1
	now = datetime.datetime.now()
	env.reset()

	for ii in range(N_SARSA):
		path = "test/CartPole-v0_SARSA_" + str(now.hour) + str(now.minute) + '_' + str(now.month) + str(now.day)
		print("Calling SARSA...")
		QL = SARSA(s_size, a_size, hneurons=[24, 24], replay_buffer=10000)
		print("SARSA learner initialized")
		rList = QL.train(env, episodes=3000, steps=201, target_update_steps=100, y=0.99, random_eps_frac=0.25, 
						 model_logging=True, folder_name=path, reward_threshold=180)
		print("done training.")
		QL.save('QCartPole_SARSA_final')
		plt.figure()
		plt.xlabel('episodes')
		plt.ylabel('rewards')
		plt.plot(rList)
		plt.savefig('QCartPole_SARSA_final.png')
		env.reset()

	# Test the trained QLearner
	runs = 10
	# rewards, num_steps = QL.simulate(env, epochs=runs, steps=100, pause=0.5, render=True)
	rewards, num_steps, _, __  = QL.simulate(env, episodes=runs, steps=100, pause=0.02, render=True)
	# rewards = np.array(rewards)
	# num_steps = np.array(num_steps)
	# sum_rewards = np.sum(rewards == 1)
	# sum_draws = np.sum(rewards == 0)
	# sum_losses = np.sum(rewards == -1)
	# print("In {} test runs, the learner won at blackjack {} percent of the time.".format(runs, (sum_rewards/runs)*100))
	# print("{} percent of games were draws.".format(100*sum_draws/runs))
	# print("{} percent of the games were losses".format(100*sum_losses/runs))
	print("In {} simulation runs, the average number of steps was {}".format(runs,np.mean(num_steps)))