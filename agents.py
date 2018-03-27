from abc import ABC, abstractmethod


class Agent(ABC):

	@abstractmethod
	def get_action(self, state):
		"""
		state: the state in which to take action
		Returns: the single action to take in this state
		"""
		pass


class KeyBoardAgent(Agent):

	def get_action(self, state):
		"""
		state: the current state from which to take action

		Returns: list of starting position, ending position
		"""

		start = [int(pos) for pos in input("Enter start position: ").split(" ")]
		end = [int(pos) for pos in input("Enter end position: ").split(" ")]

		return [start, end]