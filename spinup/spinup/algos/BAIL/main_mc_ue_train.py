import gym
import numpy as np
import torch
import argparse
import os
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.algos.BAIL import utils
from spinup.algos.ue.MC_UE import train_upper_envelope, plot_envelope

from spinup.algos.ue.models.mlp_critic import Value


def ue_train(env_set="Hopper-v2", seed=1, buffer_type="FinalSigma0.5", buffer_seed=0, buffer_size='1000K',
			 	cut_buffer_size='1000K', gamma=0.99, rollout=1000, loss_k=10000, augment_mc=True,
				max_ue_trainsteps=int(1e6), logger_kwargs=dict()):

	print('testing MClength:', rollout)
	print('Training loss ratio k:', loss_k)
	print('Discount value', gamma)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("running on device:", device)

	global logger
	logger = EpochLogger(**logger_kwargs)
	logger.save_config(locals())

	buffer_name = "%s_%s_%s" % (buffer_type, env_set, buffer_seed)
	setting_name = "%s_%s_r%s_g%s" % (buffer_name, cut_buffer_size, rollout, gamma)
	setting_name += 'noaug' if not(augment_mc) else ''
	print("---------------------------------------")
	print("Settings: " + setting_name)
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	env = gym.make(env_set)

	env.seed(seed)
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

	if not os.path.exists('./results/ueMC_%s_Gt.npy' % setting_name) :
		save_s = not os.path.exists("./results/ueMC_%s_S.npy" % (buffer_name + '_' + cut_buffer_size))
		# extract (s,a,r) pairs from replay buffer
		length = replay_buffer.get_length()
		print(length)
		states, actions, gts = [], [], []
		for ind in range(length):
			state, _, action, _, dint = replay_buffer.index(ind)
			gt =  calculate_mc_ret(replay_buffer, ind, rollout=rollout, discount=gamma) if augment_mc else \
				calculate_mc_ret_truncate(replay_buffer, ind, rollout=rollout, discount=gamma)
			gts.append(gt)
			states.append(state)
			actions.append(action)

		if save_s:
			np.save('./results/ueMC_%s_S' % (buffer_name + '_' + cut_buffer_size), states)
			np.save('./results/ueMC_%s_A' % (buffer_name + '_' + cut_buffer_size), actions)

		np.save('./results/ueMC_%s_Gt' % setting_name, gts)

	print('ue train starts ==')

	states = np.load('./results/ueMC_%s_S.npy' % (buffer_name + '_' + cut_buffer_size), allow_pickle=True)
	actions = np.load('./results/ueMC_%s_A.npy' % (buffer_name + '_' + cut_buffer_size), allow_pickle=True)
	gts = np.load('./results/ueMC_%s_Gt.npy' % setting_name, allow_pickle=True)

	upper_envelope, ue_lossval = train_upper_envelope(states, actions, gts, state_dim, device, seed, \
													  max_step_num=max_ue_trainsteps, k=loss_k, logger=logger)
	torch.save(upper_envelope.state_dict(), '%s/%s_UE.pth' % ("./pytorch_models", setting_name + \
															  '_s%s_lok%s'%(seed, loss_k)))
	print('ue train finished --')

	print('plotting ue --')

	upper_envelope = Value(state_dim, activation='relu')
	upper_envelope.load_state_dict(torch.load('%s/%s_UE.pth' % ("./pytorch_models", setting_name + \
																'_s%s_lok%s'%(seed, loss_k))))

	plot_envelope(upper_envelope, states, actions, gts, \
				  buffer_name + '_' + cut_buffer_size+'[k=%s_MClen=%s_gamma=%s'%(loss_k, rollout, gamma)+'_loss%.2f'%ue_lossval, seed)



def calculate_mc_ret(replay_buffer, idx, rollout=1000, discount=0.99):
	r_length = replay_buffer.get_length()
	_, _, _, r, d = replay_buffer.index(idx)
	mc_ret_est = r
	for h in range(1, min(rollout, r_length-idx)):
		if bool(d):  # done=True if d=1
			pass
		else:
			_, _, _, r, d = replay_buffer.index(idx + h)
			mc_ret_est += discount ** h * r

	return np.asarray(mc_ret_est)


def calculate_mc_ret_truncate(replay_buffer, idx, rollout=1000, discount=0.99):
	r_length = replay_buffer.get_length()
	state, next_state, _, r, d = replay_buffer.index(idx)
	sampled_policy_est = r
	for h in range(1, min(rollout, r_length-idx)):
		if bool(d):  # done=True if d=1
			break
		else:
			state, _, _, r, d = replay_buffer.index(idx + h)
			if (state == next_state).all():
				sampled_policy_est += discount ** h * r
				next_state = replay_buffer.index(idx + h)[1]
			else:
				break

	return np.asarray(sampled_policy_est)

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_set", default="Hopper-v2")				# OpenAI gym environment name
	parser.add_argument("--seed", default=1, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_type", default="FinalSigma0.5")				# Prepends name to filename.
	parser.add_argument("--buffer_size", default="1000K")
	parser.add_argument("--cut_buffer_size", default="1000K")
	parser.add_argument("--buffer_seed", default=0, type=int)
	parser.add_argument("--gamma", default=0.99, type=float)
	parser.add_argument("--rollout", default=1000, type=int)
	parser.add_argument("--loss_k", default=10000, type=int)
	args = parser.parse_args()

	exp_name = 'ue_mclen%s_gamma%s_k%s' % (args.rollout, args.gamma, args.loss_k)
	logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

	ue_train(env_set=args.env_set, seed=args.seed, buffer_seed=args.buffer_seed,
			 buffer_type=args.buffer_type, buffer_size=args.buffer_size, cut_buffer_size=args.cut_buffer_size,
			 gamma=args.gamma, rollout=args.rollout, loss_k=args.loss_k)

