import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor
from torch.autograd import Variable

import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(128, 128), activation='relu', init_small_weights=False, init_w=1e-3):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

        if init_small_weights:
            for affine in self.affine_layers:
                affine.weight.data.uniform_(-init_w, init_w)
                affine.bias.data.uniform_(-init_w, init_w)


    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value

# Returns an action for a given state
class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action

	
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		a = self.max_action * torch.tanh(self.l3(a)) 
		return a


class BAIL_selebah(object): # selection in mini-batch
	def __init__(self, state_dim, action_dim, max_action, max_iters, States, MCrets,
				 ue_lr=3e-3, ue_wd=2e-2, lr=1e-3, wd=0,
				 pct_anneal_type=None, last_pct=0.25,
				 select_type='border', C=None):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=wd)

		self.v_ue = Value(state_dim, activation='relu').to(device)
		self.v_ue_optimizer = torch.optim.Adam(self.v_ue.parameters(), lr=ue_lr, weight_decay=ue_wd)
		self.best_v_ue = Value(state_dim, activation='relu').to(device)
		self.ue_best_parameters = self.v_ue.state_dict()

		self.MCrets = MCrets
		test_size = int(MCrets.shape[0] * 0.2)
		self.MC_valiset_indices = np.random.randint(0, MCrets.shape[0], size=test_size)

		self.test_states = torch.from_numpy(States[self.MC_valiset_indices])
		self.test_mcrets = torch.from_numpy(self.MCrets[self.MC_valiset_indices])
		print('ue test set size:', self.test_states.size(), self.test_mcrets.size())

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.ue_valiloss_min = torch.Tensor([float('inf')]).to(device)
		self.num_increase = 0
		self.max_iters = max_iters
		self.pct_anneal_type = pct_anneal_type
		self.last_pct = last_pct
		#self.pct_info_dic = pct_info_dic
		self.select_type = select_type
		self.C = C


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, done_training_iters, iterations=5000, batch_size=1000,
			  ue_loss_k=10000, ue_vali_freq=1250,
			  logger=dict()):

		for it in range(done_training_iters, done_training_iters + iterations):

			# get batch data
			state, next_state, action, reward, done, idxs = replay_buffer.sample(batch_size, require_idxs=True)

			state = torch.FloatTensor(state).to(device)
			action = torch.FloatTensor(action).to(device)
			# next_state = torch.FloatTensor(next_state).to(device)
			# reward = torch.FloatTensor(reward).to(device)
			# done = torch.FloatTensor(1 - done).to(device)
			mc_ret = torch.FloatTensor(self.MCrets[idxs]).to(device)

			uetrain_batch_pos = [p for p, i in enumerate(idxs) if i not in self.MC_valiset_indices]
			uetrain_s = state[uetrain_batch_pos]
			uetrain_mc = mc_ret[uetrain_batch_pos]

			# train upper envelope by the k-penalty loss
			Vsi = self.v_ue(uetrain_s)
			ue_loss = L2PenaltyLoss(Vsi, uetrain_mc, k_val=ue_loss_k)

			self.v_ue_optimizer.zero_grad()
			ue_loss.backward()
			self.v_ue_optimizer.step()

			""" if it is time to recalculate border/margin """
			""" do validation for the UE network, update the best ue """
			if it % ue_vali_freq == 0:
				validation_loss = calc_ue_valiloss(self.v_ue, self.test_states, self.test_mcrets,
												   ue_bsize=int(batch_size * 0.8), ue_loss_k=ue_loss_k)

				# choose best parameters with least validation loss for the eval ue
				self.ue_valiloss_min = torch.min(self.ue_valiloss_min, validation_loss)

				if validation_loss > self.ue_valiloss_min:
					self.best_v_ue.load_state_dict(self.ue_best_parameters)
					self.num_increase += 1
				else:
					self.ue_best_parameters = self.v_ue.state_dict()
					self.num_increase = 0
				# if validation loss of ue is increasing for some consecutive steps, also return the training ue to least
				# validation loss parameters
				if self.num_increase == 4:
					self.v_ue.load_state_dict(self.ue_best_parameters)

			# estimate state values by the upper envelope
			state_value = self.best_v_ue(state).squeeze().detach()
			# project negative or small positive state values to (0, 1)
			state_value = torch.where(state_value < 1, (state_value - 1).exp(), state_value)
			if self.C is not None:
				C = self.C.to(device)
				state_value = torch.where(state_value > C, C, state_value)
			# print(type(state_value))

			# get current percentage
			if self.pct_anneal_type == 'constant':
				cur_pct = self.last_pct
			elif self.pct_anneal_type == 'linear':
				cur_pct = 1 - it / self.max_iters * (1 - self.last_pct)
			else:
				raise Exception('! undefined percentage anneal type')

			logger.store(SelePct=cur_pct)

			# determine the border / margin by current percentage
			if self.select_type == 'border':
				ratios = mc_ret / state_value
				increasing_ratios, increasing_ratio_indices = torch.sort(ratios.view(-1))
				bor_ind = increasing_ratio_indices[-int(cur_pct * batch_size)]
				border = ratios[bor_ind]

				weights = torch.where(mc_ret >= border * state_value, \
									  torch.FloatTensor([1]).to(device), torch.FloatTensor([0]).to(device))
				logger.store(Border=border.cpu().item())

			elif self.select_type == 'margin':
				diffs = mc_ret - state_value
				increasing_diffs, increasing_diff_indices = torch.sort(diffs.view(-1))
				mrg_ind = increasing_diff_indices[-int(cur_pct * batch_size)]
				margin = diffs[mrg_ind]

				weights = torch.where(mc_ret >= margin + state_value, \
									  torch.FloatTensor([1]).to(device), torch.FloatTensor([0]).to(device))
				logger.store(Margin=margin.cpu().item())

			else:
				raise Exception('! undefined selection type')

			# Compute MSE loss for actor
			update_size = weights.sum().cpu().item()
			weights = torch.stack([weights, ] * self.action_dim, dim=1)
			# print(weights.size(), action.size())
			actor_loss = torch.mul(weights, self.actor(state) - action).pow(2).mean()

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			logger.store(CloneLoss=actor_loss.detach().cpu().item(), UELoss=ue_loss.detach().cpu().item(),
						 BatchUEtrnSize=len(uetrain_batch_pos), BatchUpSize=update_size,
						 SVal=state_value.detach().mean())

		logger.store(UEValiLossMin=self.ue_valiloss_min)

		return self.best_v_ue


class BAIL_selebuf(object):  # selection in the whole buffer
	def __init__(self, state_dim, action_dim, max_action, max_iters, States, MCrets,
				 ue_lr=3e-3, ue_wd=2e-2, lr=1e-3, wd=0,
				 pct_anneal_type=None, last_pct=0.25,
				 select_type='border', C=None):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=wd)

		self.v_ue = Value(state_dim, activation='relu').to(device)
		self.v_ue_optimizer = torch.optim.Adam(self.v_ue.parameters(), lr=ue_lr, weight_decay=ue_wd)
		self.best_v_ue = Value(state_dim, activation='relu').to(device)
		self.ue_best_parameters = self.v_ue.state_dict()

		self.States = torch.from_numpy(States).float().to(device)
		self.MCrets = torch.from_numpy(MCrets).float().to(device)
		test_size = int(self.MCrets.shape[0] * 0.2)
		self.MC_valiset_indices = np.random.randint(0, MCrets.shape[0], size=test_size)
		print('ue test set size:', self.MC_valiset_indices.shape)

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.ue_valiloss_min = torch.Tensor([float('inf')]).to(device)
		self.num_increase = 0
		self.max_iters = max_iters
		self.pct_anneal_type = pct_anneal_type
		self.last_pct = last_pct
		self.pct_info_dic = pct_info_dic
		self.select_type = select_type
		self.C = C

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, done_training_iters, iterations=5000, batch_size=1000, ue_loss_k=10000,
			  ue_vali_freq=1000,
			  logger=dict()):

		for it in range(done_training_iters, done_training_iters + iterations):

			# get batch data
			state, next_state, action, reward, done, idxs = replay_buffer.sample(batch_size, require_idxs=True)

			state = torch.FloatTensor(state).to(device)
			action = torch.FloatTensor(action).to(device)
			# next_state = torch.FloatTensor(next_state).to(device)
			# reward = torch.FloatTensor(reward).to(device)
			# done = torch.FloatTensor(1 - done).to(device)
			mc_ret = self.MCrets[idxs]
			uetrain_batch_pos = [p for p, i in enumerate(idxs) if i not in self.MC_valiset_indices]
			uetrain_s = state[uetrain_batch_pos]
			uetrain_mc = mc_ret[uetrain_batch_pos]

			# train upper envelope by the k-penalty loss
			Vsi = self.v_ue(uetrain_s)
			ue_loss = L2PenaltyLoss(Vsi, uetrain_mc, k_val=ue_loss_k)

			self.v_ue_optimizer.zero_grad()
			ue_loss.backward()
			self.v_ue_optimizer.step()

			""" if it is time to recalculate border/margin """
			""" 1. calculate validation loss from the validation set, update the best ue """
			if it % ue_vali_freq == 0:

				test_states, test_mcrets = self.States[self.MC_valiset_indices], self.MCrets[self.MC_valiset_indices]
				validation_loss = calc_ue_valiloss(self.v_ue, test_states, test_mcrets,
												   ue_bsize=int(batch_size * 0.8), ue_loss_k=ue_loss_k)

				# choose the best parameters with least validation loss for the eval ue
				self.ue_valiloss_min = torch.min(self.ue_valiloss_min, validation_loss)
				logger.store(UEValiLossMin=self.ue_valiloss_min)
				if validation_loss > self.ue_valiloss_min:
					self.best_v_ue.load_state_dict(self.ue_best_parameters)
					self.num_increase += 1
				else:
					self.ue_best_parameters = self.v_ue.state_dict()
					self.num_increase = 0
				# if validation loss of ue is increasing for some consecutive steps, also return the training ue to least
				# validation loss parameters
				if self.num_increase == 4:
					self.v_ue.load_state_dict(self.ue_best_parameters)

				""" 2. estimate state values by the beat ue """
				States_values = []
				num_slice = int(np.ceil(self.States.shape[0] / batch_size))
				for i in range(num_slice):
					ind = slice(i * batch_size, min((i + 1) * batch_size, self.States.shape[0]))
					s = self.States[ind]
					States_values.append(self.best_v_ue(s).detach())
				States_values = torch.stack(States_values).view(-1)
				# States_values = self.best_v_ue(self.States).squeeze().detach()

				# project negative or small positive state values to (0, 1)
				States_values = torch.where(States_values < 1, (States_values - 1).exp(), States_values)

				if self.C is not None:
					C = self.C.to(device)
					States_values = torch.where(States_values > C, C, States_values)

				""" 3. get current percentage from the anneal plan """
				if self.pct_anneal_type == 'constant':
					cur_pct = self.last_pct
				elif self.pct_anneal_type == 'linear':
					cur_pct = 1 - it / self.max_iters * (1 - self.last_pct)
				elif self.pct_anneal_type == 'line2const':
					const_timesteps = self.pct_info_dic['const_timesteps']
					cur_pct = 1 - min(it / (self.max_iters - const_timesteps), 1.0) * (1 - self.last_pct)
				elif self.pct_anneal_type == 'convex1':  # sigmoid-shape function
					convex1_coef = self.pct_info_dic['convex1_coef']
					cur_pct = self.last_pct + 2 * (1 - self.last_pct) / (1 + np.exp(convex1_coef * it / self.max_iters))
				else:
					raise Exception('! undefined percentage anneal type')

				logger.store(SelePct=cur_pct)

				""" 4.  determine the border / margin by current percentage """
				if self.select_type == 'border':
					ratios = self.MCrets.view(-1) / States_values
					increasing_ratios, increasing_ratio_indices = torch.sort(ratios)
					bor_ind = increasing_ratio_indices[-int(cur_pct * self.States.shape[0])]
					border = ratios[bor_ind]
					logger.store(Border=border.cpu().item())
				# print(self.MCrets[increasing_ratio_indices[:100]], States_values[increasing_ratio_indices[:100]],
				#	   ratios[increasing_ratio_indices[:100]])

				elif self.select_type == 'margin':
					diffs = self.MCrets.view(-1) - States_values
					increasing_diffs, increasing_diff_indices = torch.sort(diffs)
					mrg_ind = increasing_diff_indices[-int(cur_pct * self.States.shape[0])]
					margin = diffs[mrg_ind]
					logger.store(Margin=margin.cpu().item())

				else:
					raise Exception('! undefined selection type')

			# Compute MSE loss for actor
			state_value = States_values[idxs].squeeze()
			# state_value_check = self.best_v_ue(state).squeeze().detach()
			# state_value_check = torch.where(state_value_check < 1, (state_value_check - 1).exp(), state_value_check)
			if self.select_type == 'border':
				weights = torch.where(mc_ret >= border * state_value, \
									  torch.FloatTensor([1]).to(device), torch.FloatTensor([0]).to(device))
			elif self.select_type == 'margin':
				weights = torch.where(mc_ret >= margin + state_value, \
									  torch.FloatTensor([1]).to(device), torch.FloatTensor([0]).to(device))

			update_size = weights.sum().cpu().item()
			weights = torch.stack([weights, ] * self.action_dim, dim=1)
			# print(weights.size(), action.size())
			actor_loss = torch.mul(weights, self.actor(state) - action).pow(2).mean()

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			logger.store(CloneLoss=actor_loss.detach().cpu().item(), UELoss=ue_loss.detach().cpu().item(),
						 BatchUEtrnSize=len(uetrain_batch_pos), BatchUpSize=update_size,
						 SVal=state_value.detach().mean())

		return self.best_v_ue


class BC(object):
	def __init__(self, state_dim, action_dim, max_action, lr, wd):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=wd)

		self.state_dim = state_dim

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, iterations=500, batch_size=1000, logger=dict()):
		for it in range(iterations):
			state, _, action, _, _ = replay_buffer.sample(batch_size)

			state = torch.FloatTensor(state).to(device)
			action = torch.FloatTensor(action).to(device)

			# Compute MSE loss
			actor_loss = (self.actor(state) - action).pow(2).mean()

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			logger.store(Loss=actor_loss.cpu().item())


def L2PenaltyLoss(predicted,target,k_val):
    perm = np.arange(predicted.shape[0])
    loss = Variable(torch.Tensor([0]),requires_grad=True).to(device)
    num = 0
    for i in perm:
        Vsi = predicted[i]
        yi = target[i]
        if Vsi >= yi:
            mseloss = (Vsi - yi)**2
            #loss = torch.add(loss,mseloss)
        else:
            mseloss = k_val * (yi - Vsi)**2
            num += 1
        loss = torch.add(loss, mseloss) # a very big number
    #print ('below:',num)
    return loss/predicted.shape[0]

def calc_ue_valiloss(upper_envelope, test_states, test_returns, ue_bsize, ue_loss_k):

	test_iter = int(np.ceil(test_returns.shape[0] / ue_bsize))
	validation_loss = torch.FloatTensor([0]).detach().to(device)
	for n in range(test_iter):
		ind = slice(n * ue_bsize, min((n + 1) * ue_bsize, test_returns.shape[0]))
		states_t, returns_t = test_states[ind], test_returns[ind]
		states_t = Variable(states_t.float()).to(device)
		returns_t = Variable(returns_t.float()).to(device)
		Vsi = upper_envelope(states_t)
		loss = L2PenaltyLoss(Vsi, returns_t, k_val=ue_loss_k).detach()
		validation_loss += loss

	return validation_loss


'''Training code for UE is here'''

def train_upper_envelope(states, returns, state_dim, seed,
                         upper_learning_rate=3e-3,
                         weight_decay = 0.02,
                         num_epoches = 50,
                         consecutive_steps = 4, k=10000):

	states = torch.from_numpy(np.array(states)).to(device)
	returns = torch.from_numpy(np.array(returns)).to(device)  # reward is actually returns

	# Init upper_envelope net (*use relu as activation function
	upper_envelope = Value(state_dim, activation='relu').to(device)
	upper_envelope_retrain = Value(state_dim, activation='relu').to(device)
	optimizer_upper = torch.optim.Adam(upper_envelope.parameters(), lr=upper_learning_rate,
									   weight_decay=weight_decay)


	# Split data into training and testing #
	# But make sure the highest Ri is in the training set
	# pick out the highest data point
	highestR, indice = torch.max(returns, 0)
	highestR = highestR.view(-1, 1)
	highestS = states[indice]
	print ("HighestR:",highestR)

	statesW = torch.cat((states[:indice],states[indice+1:]))
	returnsW = torch.cat((returns[:indice],returns[indice+1:]))

	# shuffle the data
	perm = np.arange(statesW.shape[0])
	np.random.shuffle(perm)
	perm = LongTensor(perm).to(device)
	statesW, returnsW = statesW[perm], returnsW[perm]

	# divide data into train/test
	divide = int(states.shape[0]*0.8)
	train_states, train_returns = statesW[:divide], returnsW[:divide]
	test_states, test_returns = statesW[divide:], returnsW[divide:]

	# add the highest data into training
	print(train_states.size(), highestS.size())
	print (train_returns.size(), highestR.size())
	train_states = torch.cat((train_states.squeeze(), highestS.unsqueeze(0)))
	train_returns = torch.cat((train_returns.squeeze(), highestR.squeeze().unsqueeze(0)))

	# train upper envelope
	# env_dummy = env_factory(0)
	# state_dim = env_dummy.observation_space.shape[0]
	# upper_envelope = Value(state_dim)
	# optimizer = torch.optim.Adam(upper_envelope.parameters(), lr=0.003, weight_decay=20)

	batch_size = 800
	optim_iter_num = int(np.ceil(train_states.shape[0] / batch_size))

	num_increase = 0
	previous_loss = float('inf')
	best_parameters = upper_envelope.state_dict()
	upper_envelope_retrain.load_state_dict(best_parameters)

	# Upper Envelope Training starts
	upper_envelope.train()

	for epoch in range(num_epoches):
		# update theta for n steps, n =
		train_loss = 0
		perm = np.arange(train_states.shape[0])
		np.random.shuffle(perm)
		perm = LongTensor(perm).to(device)

		train_states, train_returns = train_states[perm], train_returns[perm]

		for i in range(optim_iter_num):
			ind = slice(i * batch_size, min((i + 1) * batch_size, states.shape[0]))
			states_b, returns_b = train_states[ind], train_returns[ind]
			states_b = Variable(states_b.float())
			returns_b = Variable(returns_b.float())
			Vsi = upper_envelope(states_b)
			# loss = loss_fn(Vsi, returns_b)
			loss = L2PenaltyLoss(Vsi, returns_b, k_val=k)
			train_loss += loss.detach()
			upper_envelope.zero_grad()
			loss.backward()
			optimizer_upper.step()

		# early stopping

		# calculate validation error
		validation_loss = calc_ue_valiloss(upper_envelope, test_states, test_returns, ue_bsize=batch_size, ue_loss_k=k)
		if validation_loss < previous_loss:
			previous_loss = validation_loss
			best_parameters = upper_envelope.state_dict()
			upper_envelope_retrain.load_state_dict(best_parameters)
			num_increase = 0
		else:
			num_increase += 1
		if num_increase == consecutive_steps:
			upper_envelope.load_state_dict(best_parameters)

		print()
		print('Epoch:', epoch + 1)
		print('UETrainLoss:', loss.cpu().item())
		print('UEValiLoss:', validation_loss.cpu().item())

	print("Policy training is complete.")

	return upper_envelope, loss


'''Plotting code for UE is here'''

def plot_envelope(upper_envelope, states, returns, setting, seed, hyper_lst, make_title=False):

	upper_learning_rate, weight_decay, k_val, num_epoches, consecutive_steps = hyper_lst

	states = torch.from_numpy(states).to(device)
	upper_envelope = upper_envelope.to(device)
	# highestR, _ = torch.max(returns, 0)

	upper_envelope_r = []
	for i in range(states.shape[0]):
		s = states[i]
		upper_envelope_r.append(upper_envelope(s.float()).detach())

	MC_r = torch.from_numpy(returns).float().to(device)

	upper_envelope_r = torch.stack(upper_envelope_r)
	increasing_ue_vals, increasing_ue_indices = torch.sort(upper_envelope_r.view(1, -1))
	MC_r = MC_r[increasing_ue_indices[0]]

	all_ue_loss = torch.nn.functional.relu(increasing_ue_vals-MC_r).sum() + \
				  torch.nn.functional.relu(MC_r-increasing_ue_vals).sum()*k_val

	plt.rc('legend', fontsize=14)  # legend fontsize
	fig, axs = plt.subplots()

	axs.set_xlabel('state', fontsize=28)
	axs.set_ylabel('Returns and \n Upper Envelope', fontsize=28, multialignment="center")

	plot_s = list(np.arange(states.shape[0]))
	plt.scatter(plot_s, list(MC_r.view(1, -1).cpu().numpy()[0]), s=0.5, color='orange', label='MC Returns')
	plt.plot(plot_s, list(increasing_ue_vals.view(1, -1).cpu().numpy()[0]), color='blue', label="Upper Envelope")

	ue_info = '_loss_%.2fe6' % (all_ue_loss.item()/1e6)
	if make_title:
		title = setting.replace('_r', '\nr') + ue_info
		plt.title(title)
	plt.legend()
	plt.xticks(fontsize=18, rotation=15)
	plt.yticks(fontsize=18)
	plt.tight_layout()
	plt.savefig('./plots/' + "ue_visual_%s.png" % setting)
	plt.close('all')

	print('Plotted current UE in', "ue_visual_%s.png" % setting)

	return


def plot_envelope_with_clipping(upper_envelope, states, returns, setting, seed, hyper_lst, make_title=False, S=10000):

	upper_learning_rate, weight_decay, k_val, num_epoches, consecutive_steps = hyper_lst

	states = torch.from_numpy(states).to(device)
	upper_envelope = upper_envelope.to(device)

	plt.rc('legend', fontsize=14)  # legend fontsize
	fig, axs = plt.subplots()

	axs.set_xlabel('state', fontsize=28)
	axs.set_ylabel('Returns and \n Upper Envelope', fontsize=28, multialignment="center")

	upper_envelope_r = []
	for i in range(states.shape[0]):
		s = states[i]
		upper_envelope_r.append(upper_envelope(s.float()).detach())

	MC_r = torch.from_numpy(returns).float().to(device)

	upper_envelope_r = torch.stack(upper_envelope_r)
	increasing_ue_returns, increasing_ue_indices = torch.sort(upper_envelope_r.view(1, -1))
	MC_r = MC_r[increasing_ue_indices[0]]

	# Do auto clipping
	perm = np.arange(states.shape[0])
	Diff = []
	for idx in perm:
		Diff.append(increasing_ue_returns[0, idx] - MC_r.view(1, -1).cpu().numpy()[0, idx])

	eval_point = states.shape[0] - 1
	Clipping_value = increasing_ue_returns[0, eval_point]
	while eval_point >= S:
		min_Diff = min(Diff[eval_point - S:eval_point])
		if min_Diff < 0:
			Clipping_value = increasing_ue_returns[0, eval_point]
			break
		eval_point -= S

	Adapt_Clip = []
	for i in range(states.shape[0]):
		Adapt_Clip.append(Clipping_value)
	Adapt_Clip = torch.FloatTensor(Adapt_Clip).to(device)

	clipped_ue_r = torch.where(increasing_ue_returns > Adapt_Clip, Adapt_Clip, increasing_ue_returns)
	#num_above = torch.where(clipped_ue_r > MC_r, torch.FloatTensor([1]), torch.FloatTensor([0])).sum().item()
	Clipping_loss = F.relu(clipped_ue_r-MC_r).sum() + F.relu(MC_r-clipped_ue_r).sum()*k_val

	plot_s = list(np.arange(states.shape[0]))
	plt.scatter(plot_s, list(MC_r.view(1, -1).cpu().numpy()[0]), s=0.5, color='orange', label='MC Returns')
	plt.plot(plot_s, list(increasing_ue_returns.view(1, -1).cpu().numpy()[0]), color='blue', label="Upper Envelope")
	plt.plot(plot_s, Adapt_Clip.cpu().numpy(), color='black', label="Adaptive_Clipping_%s" % eval_point)
	clip_info = '_clip_%.2f_loss_%.2fe6_ues_%s' % (Clipping_value.item(), Clipping_loss.item()/1e6, seed)
	if make_title:
		plt.title(setting.replace('K', 'K\n'))
	plt.legend()
	plt.tight_layout()
	plt.savefig('./plots/' + "ue_visual_%s_Clipped.png" % (setting + clip_info))
	plt.close('all')

	print('Plotting finished')

	return Clipping_value

