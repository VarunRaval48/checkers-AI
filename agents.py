from abc import ABC, abstractmethod


class Agent(ABC):

	def __init__(self, is_learning_agent=False):
		self.is_learning_agent = is_learning_agent


	@abstractmethod
	def get_action(self, state):
		"""
		state: the state in which to take action
		Returns: the single action to take in this state
		"""
		pass


class KeyBoardAgent(Agent):

	def __init__(self):
		Agent.__init__(self)


	def get_action(self, state):
		"""
		state: the current state from which to take action

		Returns: list of starting position, ending position
		"""

		start = [int(pos) for pos in input("Enter start position (e.g. x y): ").split(" ")]
		end = [int(pos) for pos in input("Enter end position (e.g. x y): ").split(" ")]

		return [start, end]


"""
class AlphaBetaAgent(Agent):

	def __init__(self):
		Agent.__init__(self)

	def get_action(self, state):

"""


class RinforcementLearningAgent(Agent):

	def __init__(self, num_training=10):
		Agent.__init__(self, True)

		self.num_training = num_training
		self.episodes_so_far = 0


	@abstractmethod
	def get_action(self, state):
		"""
		state: the current state from which to take action

		Returns: the action to perform
		"""
		# TODO call do_action from this method
		pass


	@abstractmethod
	def update(self, state, action, reward, next_state):
		"""
		performs update for the learning agent

		state: the state (s) in which action was taken
		action: the action (a) taken in the state (s)
		reward: reward obtained for taking action (a) in state (s) and going to next state (s')
		next_state: the next state (s'), in which agnet will perform next action, 
					that resulted from state (s) and action (a)
		"""
		pass


	def observe_transition(self, state, action, reward, next_state):
		"""
		state: the state (s) in which action was taken
		action: the action (a) taken in the state (s)
		reward: reward obtained for taking action (a) in state (s) and going to next state (s')
		next_state: the next state (s'), in which agnet will perform next action, 
					that resulted from state (s) and action (a)
		"""
		self.episode_rewards += reward
		self.update(state, action, reward, next_state)


	def start_episode(self):
		"""
		called by environment to notify agent of starting new episode
		"""

		self.prev_state = None
		self.prev_action = None
		self.episode_rewards = 0.0


	def stop_episode(self):
		"""
		called by environment to notify agent about end of episode
		"""

		self.episodes_so_far += 1


	# TODO
	def reward_function(self, state, action, next_state):
		pass


	def do_action(self, state, action):
		"""
		called by get_action to update previous state and action
		"""
		self.prev_state = state
		self.prev_action = action


	def observation_function(self, state):
		if self.prev_state is not None:

			reward = reward_function(self.prev_state, self.prev_action, state)
			self.observe_transition(self.prev_state, self.prev_action, reward, state)