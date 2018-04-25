"""
This file contains methods to interpret the statistics formed during training.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

def plot_train(file_name):
	result = np.loadtxt(file_name, skiprows=1, ndmin=2, delimiter=',')

	num_moves = result[:, 0]
	wins = result[:, 1]
	rewards = result[:, 2]
	max_q_values = result[:, 3]

	num_games = len(result)

	xlabel = 'number of game'

	plt.title('Number of moves vs episodes')
	plt.plot(range(0, num_games), num_moves, 'o', markersize=5)
	plt.xlabel(xlabel)
	plt.ylabel('Number of moves')
	plt.show()

	figure = plt.figure(figsize=(5, 5))
	plt.title('Wins vs episodes')
	plt.plot(range(0, num_games), wins, 'o', markersize=5)
	plt.ylabel('Wins or loss')
	plt.xlabel(xlabel)
	plt.show()

	figure = plt.figure(figsize=(5, 5))
	plt.title('Rewards vs episodes')
	plt.plot(range(0, num_games), rewards, 'o', markersize=5)
	plt.ylabel('Rewards')
	plt.xlabel(xlabel)
	plt.show()

	figure = plt.figure(figsize=(5, 5))
	plt.title('Max Q Values vs episodes')
	plt.plot(range(0, num_games), max_q_values, '-o', markersize=5)
	plt.ylabel('Max Q values')
	plt.xlabel(xlabel)
	plt.show()


if __name__ == '__main__':

	args = sys.argv[1]
	plot_train(args)
