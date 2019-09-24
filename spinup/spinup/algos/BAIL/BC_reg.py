import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from spinup.algos.BAIL import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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



class BC_reg(object):
	def __init__(self, state_dim, action_dim, max_action, lr, wd):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=wd)

		self.state_dim = state_dim


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, iterations=500, batch_size=100, logger=dict()):

		for it in range(iterations):

			state, next_state, action, reward, done = replay_buffer.sample(batch_size)

			state = torch.FloatTensor(state).to(device)
			action = torch.FloatTensor(action).to(device)
			next_state = torch.FloatTensor(next_state).to(device)
			reward = torch.FloatTensor(reward).to(device)
			done = torch.FloatTensor(1 - done).to(device)


			# Compute MSE loss
			actor_loss = (self.actor(state) - action).pow(2).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			logger.store(Loss=actor_loss.cpu().item())




