from abc import ABC, abstractmethod
from util import *
import random
from game import CHECKERS_FEATURE_COUNT, checkers_features


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


class ReinforcementLearningAgent(Agent):

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
	def update(self, state, action, next_state, reward):
		"""
		performs update for the learning agent

		state: the state (s) in which action was taken
		action: the action (a) taken in the state (s)
		next_state: the next state (s'), in which agnet will perform next action, 
					that resulted from state (s) and action (a)
		reward: reward obtained for taking action (a) in state (s) and going to next state (s')
		"""
		pass


	def observe_transition(self, state, action, next_state, reward):
		"""
		state: the state (s) in which action was taken
		action: the action (a) taken in the state (s)
		next_state: the next state (s'), in which agnet will perform next action, 
					that resulted from state (s) and action (a)
		reward: reward obtained for taking action (a) in state (s) and going to next state (s')
		"""
		self.episode_rewards += reward
		self.update(state, action, next_state, reward)


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

			reward = self.reward_function(self.prev_state, self.prev_action, state)
			self.observe_transition(self.prev_state, self.prev_action, state, reward)



class QlearningAgent(ReinforcementLearningAgent):

	def __init__(self, alpha=0.9, gamma=0.1, epsilon=0.5, num_training=10):
		ReinforcementLearningAgent.__init__(self, num_training=num_training)

		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon

		# initialize weights for the features
		self.weights = [0]*CHECKERS_FEATURE_COUNT


    def get_q_value(self, state, action, features):
        """
          Returns: Q(state,action)
        """
        q_value = 0
        for i in range(CHECKERS_FEATURE_COUNT):
        	q_value += self.weights[i] * features[i]

        return q_value


    def compute_value_from_q_values(self, state):
        """
          Returns: max_action Q(state, action) where the max is over legal actions.
          		   If there are no legal actions, which is the case at the terminal state, 
          		   return a value of 0.0.
        """
        actions = state.get_legal_actions()

        if not actions:
            return 0.0
        
        max_value = None
        for i, action in enumerate(actions):
            temp_q_value = self.get_q_value(state, action, checkers_features(state, action))
            if max_value is None or max_value < temp_q_value:
                max_value = temp_q_value

        return max_value


    def compute_action_from_q_values(self, state, actions):
        """
          Returns: the best action to take in a state. If there are no legal actions,
          		   which is the case at the terminal state, return None.
        """
        if not actions:
            return None
        
        max_value = None
        arg_max = None
        for i, action in enumerate(actions):
            temp_q_value = self.get_q_value(state, action, checkers_features(state, action))
            if max_value is None or max_value < temp_q_value:
                max_value = temp_q_value
                arg_max = i

        # if max_value < 0:
        #     return random.choice(actions)

        return actions[arg_max]


    def get_action(self, state):
        """
          Returns: the action to take in the current state.  With probability self.epsilon,
		           take a random action and take the best policy action otherwise.  If there are
          		   no legal actions, which is the case at the terminal state, returns None.
        """

        # Pick Action
        legal_actions = state.get_legal_actions()
        action = None

        if not legal_actions:
            return None

        if util.flip_coin(self.epsilon):
            action = random.choice(legal_actions)
        else:
            action = self.compute_action_from_q_values(state, legal_actions)

        self.do_action(state, action)
        return action


    def update(self, state, action, next_state, reward):

        features = checkers_features(state, action)

        expected = reward + self.gamma * self.compute_value_from_q_values(next_state)
        current = self.get_q_value(state, action, features)

        update = expected - current

        for i in range(CHECKERS_FEATURE_COUNT):
        	self.weights[i] = self.weights[i] + self.alpha * (update) * features



    def getPolicy(self, state):
        return self.compute_action_from_q_values(state, state.get_legal_actions())


    def getValue(self, state):
        return self.compute_value_from_q_values(state)	