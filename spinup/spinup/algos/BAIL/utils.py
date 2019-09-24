import numpy as np

# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py


'''SARS replay buffer'''
class ReplayBuffer(object):
	def __init__(self):
		self.storage = []

	# Expects tuples of (state, next_state, action, reward, done)
	def add(self, data):
		self.storage.append(data)

	def sample(self, batch_size, require_idxs=False, space_rollout=0):
		ind = np.random.randint(0, len(self.storage) - space_rollout,
								size=batch_size)
		state, next_state, action, reward, done = [], [], [], [], []

		for i in ind: 
			s, s2, a, r, d = self.storage[i]
			state.append(np.array(s, copy=False))
			next_state.append(np.array(s2, copy=False))
			action.append(np.array(a, copy=False))
			reward.append(np.array(r, copy=False))
			done.append(np.array(d, copy=False))

		if require_idxs:
			return (np.array(state),
					np.array(next_state),
					np.array(action),
					np.array(reward).reshape(-1, 1),
					np.array(done).reshape(-1, 1), ind)
		else:
			return (np.array(state),
					np.array(next_state),
					np.array(action),
					np.array(reward).reshape(-1, 1),
					np.array(done).reshape(-1, 1))

	def index (self, i):
		return self.storage[i]

	def save(self, filename):
		np.save("./buffers/"+filename+"sars.npy", self.storage)

	def load(self, filename):
		self.storage = np.load("./buffers/"+filename+"sars.npy",
                                allow_pickle=True)

	def cut_final(self, buffer_size):
		self.storage = self.storage[ -int(buffer_size): ]

	def get_length(self):
		return self.storage.__len__()



'''SARSA replay buffer'''
class SARSAReplayBuffer(object):
	def __init__(self):
		self.storage = []

	# Expects tuples of (state, next_state, action, reward, done)
	def add(self, data):
		self.storage.append(data)

	def sample(self, batch_size, require_idxs=False, space_rollout=0):
		ind = np.random.randint(0, len(self.storage)-space_rollout,
                                        size=batch_size)
		state, next_state, action, next_action, reward, done = [], [], [], [], [], []

		for i in ind:
			s, s2, a, a2, r, d = self.storage[i]
			state.append(np.array(s, copy=False))
			next_state.append(np.array(s2, copy=False))
			action.append(np.array(a, copy=False))
			next_action.append(np.array(a2, copy=False))
			reward.append(np.array(r, copy=False))
			done.append(np.array(d, copy=False))
		
		if require_idxs:
			return (np.array(state),
					np.array(next_state),
					np.array(action),
					np.array(next_action),
					np.array(reward).reshape(-1, 1),
					np.array(done).reshape(-1, 1), ind)
		else:
			return (np.array(state),
					np.array(next_state),
					np.array(action),
					np.array(next_action),
					np.array(reward).reshape(-1, 1),
					np.array(done).reshape(-1, 1))

	def index (self, i):
		return self.storage[i]

	def save(self, filename):
		np.save("./buffers/"+filename+"_sarsa.npy", self.storage)

	def load(self, filename):
		self.storage = np.load("./buffers/"+filename+"_sarsa.npy",
                                       allow_pickle=True)

	def cut_final(self, buffer_size):
		self.storage = self.storage[ -int(buffer_size): ]

	def get_length(self):
		return self.storage.__len__()
