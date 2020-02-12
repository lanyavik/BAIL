import gym
import numpy as np
import torch
import argparse
import os

from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.algos.BAIL import utils, bail_training
from spinup.algos.BAIL.bail_training import Value, train_upper_envelope, plot_envelope, plot_envelope_with_clipping

# check directory
print('data directory', os.getcwd())
# check pytorch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on device:", device)

def bail_learn(env_set="Hopper-v2", seed=0, buffer_type="FinalSigma0.5_env_0_1000K",
					gamma=0.99, ue_rollout=1000, augment_mc='gain',
					ue_lr=3e-3, ue_wd=2e-2, ue_loss_k=1000, ue_train_epoch=50,
					clip_ue=False, detect_interval=10000,
			    	eval_freq=500, max_timesteps=int(2e5), batch_size=int(1e3), lr=1e-3, wd=0, pct=0.3,
			    	logger_kwargs=dict()):


	"""set up logger"""
	global logger
	logger = EpochLogger(**logger_kwargs)
	logger.save_config(locals())


	file_name = "bail_stat_%s_%s" % (env_set, seed)
	setting_name = "%s_r%s_g%s" % (buffer_type.replace('env', env_set), ue_rollout, gamma)
	setting_name += '_noaug' if not (augment_mc) else ''
	setting_name += '_augNew' if augment_mc == 'new' else ''
	#in oracle study, use gain calculated via Oracle data, but use NonOracle data to train UE and policy
	if 'Oracle' == buffer_type[:6]:
		buffer_type = 'Non' + buffer_type

	print("---------------------------------------")
	print("Algo: " + file_name + "\tData: " + buffer_type)
	print("Settings: " + setting_name)
	print("Evaluate Policy every", eval_freq * batch_size / 1e6,
		  'epoches; Total', max_timesteps* batch_size / 1e6, 'epoches')
	print("---------------------------------------")

	env = gym.make(env_set)
	test_env = gym.make(env_set)

	# Set seeds
	env.seed(seed)
	test_env.seed(seed)
	env.action_space.np_random.seed(seed)
	test_env.action_space.np_random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	# Load buffer
	replay_buffer = utils.ReplayBuffer()
	buffer_name = buffer_type.replace('env', env_set)
	replay_buffer.load(buffer_name)

	states = np.load('./results/ueMC_%s_S.npy' % buffer_name, allow_pickle=True).squeeze()
	setting_name  += '_Gain' if augment_mc == 'gain' else '_Gt'
	returns = np.load('./results/ueMC_%s.npy' % setting_name, allow_pickle=True).squeeze()
	print('Load mc returns type', augment_mc, 'with gamma:', gamma, 'rollout length:', ue_rollout)

	cur_ue_setting = 'Stat_' + setting_name + '_lossk%s_s%s' % (ue_loss_k, seed)

	if not os.path.exists('%s/Stat_UE_%s.pth' % ("./pytorch_models", setting_name + '_s%s_lok%s' % (seed, ue_loss_k))):
		# train ue
		print('ue train starts --')
		print('with testing MClength:', ue_rollout, 'training loss ratio k:', ue_loss_k)
		upper_envelope, _ = train_upper_envelope(states, returns, state_dim, seed, upper_learning_rate=ue_lr,
														weight_decay = ue_wd, num_epoches = ue_train_epoch, k=ue_loss_k)
		torch.save(upper_envelope.state_dict(), '%s/Stat_UE_%s.pth' % ("./pytorch_models", setting_name + \
																  '_s%s_lok%s' % (seed, ue_loss_k)))
		print('plotting ue --')
		plot_envelope(upper_envelope, states, returns, cur_ue_setting, seed, [ue_lr, ue_wd, ue_loss_k, ue_train_epoch, 4])

	else:
		upper_envelope = Value(state_dim, activation='relu')
		upper_envelope.load_state_dict(torch.load('%s/Stat_UE_%s.pth' % ("./pytorch_models", setting_name + '_s%s_lok%s' % (seed, ue_loss_k))))
		print('Load seed %s envelope from'%seed, 'with training loss ratio k:', ue_loss_k)

	# do clipping if needed
	C = plot_envelope_with_clipping(upper_envelope, states, returns, cur_ue_setting, seed,
								  [ue_lr, ue_wd, ue_loss_k, max_timesteps, 4], S=detect_interval) if clip_ue else None
	print('clipping at:', C)


	print('Doing selection in Buffer via ue --')
	selected_buffer, selected_len, border = select_batch_ue(replay_buffer, states, returns, upper_envelope, seed, ue_loss_k,\
								   							C, select_percentage=pct)

	print('-- Policy train starts --')
	# Initialize policy
	policy = bail_training.BC(state_dim, action_dim, max_action, lr=lr, wd=wd)

	training_iters, epoch = 0, ue_train_epoch
	while training_iters < max_timesteps:
		epoch += eval_freq * batch_size / 1e6
		pol_vals = policy.train(selected_buffer, iterations=int(eval_freq), batch_size=batch_size, logger=logger)

		avgtest_reward = evaluate_policy(policy, test_env)
		training_iters += eval_freq


		logger.log_tabular('Epoch', epoch)
		logger.log_tabular('AverageTestEpRet', avgtest_reward)
		logger.log_tabular('TotalSteps', training_iters)
		logger.log_tabular('Loss', average_only=True)
		logger.log_tabular('SelectSize', selected_len)
		logger.log_tabular('Border', border.item())

		logger.dump_tabular()



def select_batch_ue(replay_buffer, states, returns, upper_envelope, seed, ue_loss_k, C, select_percentage):

	states = torch.from_numpy(states).to(device)
	returns = torch.from_numpy(returns).to(device)
	upper_envelope = upper_envelope.to(device)

	ratios = []
	for i in range(states.shape[0]):
		s, ret = states[i], returns[i]
		s_val = upper_envelope(s.unsqueeze(dim=0).float()).detach().squeeze()
		ratios.append(ret / torch.min(s_val, C) if C is not None else ret / s_val)

	ratios = torch.stack(ratios).view(-1)
	increasing_ratios, increasing_ratio_indices = torch.sort(ratios)
	bor_ind = increasing_ratio_indices[-int(select_percentage*states.shape[0])]
	border = ratios[bor_ind]

	'''begin selection'''
	selected_buffer = utils.ReplayBuffer()
	print('Selecting with ue border', border.item())
	for i in range(states.shape[0]):
		rat = ratios[i]
		if rat >= border:
			obs, _, act, _, _  = replay_buffer.index(i)
			selected_buffer.add((obs, None, act, None, None))

	initial_len, selected_len = replay_buffer.get_length(), selected_buffer.get_length()
	print('border:', border, 'selecting ratio:', selected_len, '/', initial_len)

	return (selected_buffer, selected_len, border)


# Runs policy for X episodes and returns average reward
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

	print ("---------------------------------------")
	print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
	print ("---------------------------------------")
	return avg_reward



if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_set", default="Hopper-v2")  # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=int(5e2), type=int)  # How often (time steps) we evaluate
	parser.add_argument("--ue_train_epoch", default=50, type=int)
	parser.add_argument("--max_timesteps", default=int(2e5), type=int)  # Max time steps to run environment for
	parser.add_argument('--exp_name', type=str, default='bail_stat_local')
	args = parser.parse_args()

	logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

	bail_learn(env_set=args.env_set, seed=args.seed,
                eval_freq=args.eval_freq, max_timesteps=args.max_timesteps, ue_train_epoch=args.ue_train_epoch,
                logger_kwargs=logger_kwargs)
