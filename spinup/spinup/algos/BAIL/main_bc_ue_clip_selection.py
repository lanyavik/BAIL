import gym
import numpy as np
import torch
import argparse
import os

from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.algos.BAIL import utils, BC_reg
from spinup.algos.ue.MC_UE import plot_envelope_with_clipping

from spinup.algos.ue.models.mlp_critic import Value


def bc_ue_learn(env_set="Hopper-v2", seed=0, buffer_type="FinalSigma0.5", buffer_seed=0, buffer_size='1000K',
                cut_buffer_size='1000K',
				ue_seed_list=[1, 2, 3, 4, 5], gamma=0.99, ue_rollout=1000, ue_loss_k=10000, augment_mc=True,
				clip_ue="f-auto", detect_interval=10000, k_prime=10000,
			    eval_freq=float(500), max_timesteps=float(1e5), lr=1e-3, wd=0, P=0.25,
			    logger_kwargs=dict()):

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("running on device:", device)

	"""set up logger"""
	global logger
	logger = EpochLogger(**logger_kwargs)
	logger.save_config(locals())


	file_name = "BCueclip_%s_%s" % (env_set, seed)
	buffer_name = "%s_%s_%s" % (buffer_type, env_set, buffer_seed)
	setting_name = "%s_%s_r%s_g%s" % (buffer_name, cut_buffer_size, ue_rollout, gamma)
	setting_name += 'noaug' if not (augment_mc) else ''

	print
	("---------------------------------------")
	print
	("Settings: " + file_name)
	print
	("---------------------------------------")


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
	replay_buffer.load(buffer_name + '_' + buffer_size)
	if buffer_size != cut_buffer_size:
		replay_buffer.cut_final(int(cut_buffer_size[:-1]) * 1e3)
	print(replay_buffer.get_length())

	print('buffer setting:', buffer_name + '_' + cut_buffer_size)

	print('clip and selection type:', clip_ue)
	if clip_ue is None:
		best_ue_seed = ue_seed_list
		C = None
	elif clip_ue == "s-auto":
		best_ue_seed = ue_seed_list
		print('-- Do clipping on the selected envelope --')
		C, _ = get_ue_clipping_info(best_ue_seed, ue_loss_k, k_prime, detect_interval, setting_name, state_dim,\
			buffer_info=buffer_name + '_' + cut_buffer_size, ue_setting='[k=%s_MClen=%s_gamma=%s'%(ue_loss_k, ue_rollout, gamma))
	elif clip_ue == "f-auto":
		print('-- Do clipping on each envelope --')
		ues_info = dict()
		for ue_seed in ue_seed_list:
			ues_info[ue_seed] = get_ue_clipping_info(ue_seed, ue_loss_k, k_prime, detect_interval, setting_name, state_dim,\
			buffer_info=buffer_name + '_' + cut_buffer_size, ue_setting='[k=%s_MClen=%s_gamma=%s'%(ue_loss_k, ue_rollout, gamma))
		print('Auto clipping info:', ues_info)
		clipping_val_list, clipping_loss_list = tuple(map(list, zip(*ues_info.values())))
		sele_idx = int(np.argmin(np.array(clipping_loss_list)))
		best_ue_seed = ue_seed_list[sele_idx]
		C = clipping_val_list[sele_idx]


	print("Best UE", best_ue_seed, "Clipping value: ", C)


	gts = np.load('./results/ueMC_%s_Gt.npy' % setting_name, allow_pickle=True)
	print('Load gts from', './results/ueMC_%s_Gt.npy' % setting_name)
	upper_envelope = Value(state_dim, activation='relu')
	upper_envelope.load_state_dict(torch.load('%s/%s_UE.pth' % ("./pytorch_models", setting_name+'_s%s_lok%s'%(best_ue_seed, ue_loss_k))))
	print('Load best envelope from', '%s/%s_UE.pth' % ("./pytorch_models", setting_name+'_s%s_lok%s'%(best_ue_seed, ue_loss_k)))
	print('with testing MClength:', ue_rollout, 'training loss ratio k:', ue_loss_k, 'clipping at', C)

	selected_buffer, selected_len, border = select_batch_ue(replay_buffer, setting_name, buffer_name + '_' + cut_buffer_size,
															state_dim, best_ue_seed, ue_loss_k,
								   							C, select_percentage=P)

	print('-- Policy train starts --')
	# Initialize policy
	policy = BC_reg.BC_reg(state_dim, action_dim, max_action, lr=lr, wd=wd)

	episode_num = 0
	done = True

	training_iters, epoch = 0, 0
	while training_iters < max_timesteps:
		epoch += 1
		pol_vals = policy.train(selected_buffer, iterations=int(eval_freq), logger=logger)

		avgtest_reward = evaluate_policy(policy, test_env)
		training_iters += eval_freq


		logger.log_tabular('Epoch', epoch)
		logger.log_tabular('AverageTestEpRet', avgtest_reward)
		logger.log_tabular('TotalSteps', training_iters)
		logger.log_tabular('Loss', average_only=True)
		logger.log_tabular('UESeed', best_ue_seed)
		logger.log_tabular('SelectSize', selected_len)
		logger.log_tabular('Border', border)

		logger.dump_tabular()


def get_ue_clipping_info(ue_seed, ue_loss_k, k_prime, detect_interval, setting_name, state_dim, buffer_info, ue_setting):

	states = np.load('./results/ueMC_%s_S.npy' % buffer_info, allow_pickle=True)
	returns = np.load('./results/ueMC_%s_Gt.npy' % setting_name, allow_pickle=True)
	upper_envelope = Value(state_dim, activation='relu')
	upper_envelope.load_state_dict(
		torch.load('%s/%s_UE.pth' % ("./pytorch_models", setting_name + '_s%s_lok%s' % (ue_seed, ue_loss_k))))

	clipping_val, clipping_loss = plot_envelope_with_clipping(upper_envelope, states, returns, buffer_info+ue_setting, ue_seed,
								  hyper_default=True, k_prime=k_prime, S=detect_interval)

	return clipping_val, clipping_loss


def select_batch_ue(replay_buffer, setting_name, buffer_info, state_dim, best_ue_seed, ue_loss_k, C, select_percentage):

	states = np.load('./results/ueMC_%s_S.npy' % buffer_info, allow_pickle=True)
	returns = np.load('./results/ueMC_%s_Gt.npy' % setting_name, allow_pickle=True)
	upper_envelope = Value(state_dim, activation='relu')
	upper_envelope.load_state_dict(
		torch.load('%s/%s_UE.pth' % ("./pytorch_models", setting_name + '_s%s_lok%s' % (best_ue_seed, ue_loss_k))))


	ratios = []
	for i in range(states.shape[0]):
		s, gt = torch.FloatTensor([states[i]]), torch.FloatTensor([returns[i]])
		s_val = upper_envelope(s.unsqueeze(dim=0).float()).detach().squeeze()
		ratios.append(gt / torch.min(s_val, C) if C is not None else gt / s_val)
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
			data = replay_buffer.index(i)
			selected_buffer.add(data)

	initial_len, selected_len = replay_buffer.get_length(), selected_buffer.get_length()
	print(selected_len, '/', initial_len, 'selecting ratio:', selected_len/initial_len)

	selection_info = 'ue_C%.2f' % C if C is not None else 'ue_none'
	selection_info += '_bor%.2f_len%s' % (border, selected_len)
	selected_buffer.save(selection_info +'_'+ buffer_info)

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
	parser.add_argument("--env_name", default="Hopper-v2")				# OpenAI gym environment name
	parser.add_argument("--seed", default=1, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_type", default="FinalSigma0.5")				# Prepends name to filename.
	parser.add_argument("--eval_freq", default=1e2, type=float)			# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=float)		# Max time steps to run environment for
	parser.add_argument('--exp_name', type=str, default='bc_ue_b')
	args = parser.parse_args()

	logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

	bc_ue_learn(env_set=args.env_name, seed=args.seed, buffer_type=args.buffer_type,
                eval_freq=args.eval_freq,
                max_timesteps=args.max_timesteps,
                logger_kwargs=logger_kwargs)
