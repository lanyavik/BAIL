import numpy as np
import torch
import gym
import argparse
import os
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.algos.BAIL import utils, DDPG_col


def ddpg_genbuf(env_set="Hopper-v2", seed=0, max_timesteps=float(1e6), start_timesteps=int(1e3),
				expl_noise=0.5,
			    eval_freq='episode_timesteps',
			    logger_kwargs=dict()):


	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("running on device:", device)

	"""set up logger"""
	global logger
	logger = EpochLogger(**logger_kwargs)
	logger.save_config(locals())


	file_name = "DDPG_%s_%s" % (env_set, str(seed))
	buffer_name = "FinalSigma%s_%s_%s_%sK" % (str(expl_noise), env_set, str(seed),
										   str(int(max_timesteps/1e3)))
	exp_name = "ddpg_collection_%s_steps%s_sigma%s_%s" \
			   % (env_set, str(max_timesteps), str(expl_noise), str(seed))
	print ("---------------------------------------")
	print ("Settings: " + file_name)
	print ("Save Buffer as: " + buffer_name)
	print ("---------------------------------------")

	if not os.path.exists("./pytorch_models"):
		os.makedirs("./pytorch_models")


	env = gym.make(env_set)
	test_env = gym.make(env_set)

	# Set seeds
	'''for algos with environment interacts we also have to seed env.action_space'''
	env.seed(seed)
	test_env.seed(seed)
	env.action_space.np_random.seed(seed)
	test_env.action_space.np_random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])
	print('max episode length', env._max_episode_steps)

	# Initialize policy and buffer
	policy = DDPG_col.DDPG(state_dim, action_dim, max_action)
	replay_buffer = utils.ReplayBuffer()
	
	total_timesteps = 0
	episode_num = 0
	done = True 

	while total_timesteps < max_timesteps:
		
		if done: 

			if total_timesteps != 0:
				policy.train(replay_buffer, episode_timesteps)

				avgtest_reward = evaluate_policy(policy, test_env, eval_episodes=10)


				logger.log_tabular('Episode', episode_num)
				logger.log_tabular('AverageTestEpRet', avgtest_reward)
				logger.log_tabular('TotalSteps', total_timesteps)
				logger.log_tabular('EpRet', episode_reward)
				logger.log_tabular('EpLen', episode_timesteps)
				logger.dump_tabular()


			# Reset environment
			obs = env.reset()
			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 
		
		# Select action randomly or according to policy
		if total_timesteps < start_timesteps:
			action = env.action_space.sample()
		else:
			action = policy.select_action(np.array(obs))
			if expl_noise != 0:
				action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape[0]))\
							  .clip(env.action_space.low, env.action_space.high)

		# Perform new action!!!
		new_obs, reward, done, _ = env.step(action)
		episode_reward += reward
		episode_timesteps += 1
		total_timesteps += 1

		done_bool = 0 if episode_timesteps == env._max_episode_steps else float(done)

		# Store data in replay buffer

		replay_buffer.add((obs, new_obs, action, reward, done_bool))
		obs = new_obs

	# Save final policy
	policy.save("%s" % (file_name), directory="./pytorch_models")
	# Save final buffer
	replay_buffer.save(buffer_name)

def evaluate_policy(policy, env, eval_episodes=10):
	tol_reward = 0
	for _ in range(eval_episodes):
		obs = env.reset()
		done = False
		while not done:
			action = policy.select_action(np.array(obs))
			obs, reward, done, _ = env.step(action)
			tol_reward += reward

	avg_reward = tol_reward / eval_episodes

	return avg_reward


# Shortened version of code originally found at https://github.com/sfujim/TD3
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--env_set", default="Hopper-v2")  # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
	parser.add_argument("--start_timesteps", default=1e3, type=int)  # How many time steps purely random policy is run for
	parser.add_argument("--expl_noise", default=0.5, type=float)  # Std of Gaussian exploration noise
	args = parser.parse_args()

	exp_name='ddpgcol_sigma%s_trainlen%s_%s'%(args.expl_noise, args.max_timesteps, args.env_set)
	logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

	ddpg_genbuf(env_set=args.env_set, seed=args.seed,
				max_timesteps=args.max_timesteps, start_timesteps=args.start_timesteps,
				expl_noise=args.expl_noise,
				logger_kwargs=logger_kwargs)

