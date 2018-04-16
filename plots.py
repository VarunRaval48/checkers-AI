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

	figure = plt.figure()
	xlabel = 'number of episode'

	plt.title('Number of moves vs episodes')
	plt.plot(range(0, num_games), num_moves, '-o', markersize=5)
	plt.xlabel(xlabel)
	plt.ylabel('Number of moves')
	plt.show()

	plt.title('Wins vs episodes')
	plt.plot(range(0, num_games), wins, '-o', markersize=5)
	plt.ylabel('Wins or loss')
	plt.show()

	plt.title('Rewards vs episodes')
	plt.plot(range(0, num_games), rewards, '-o', markersize=5)
	plt.ylabel('Rewards')
	plt.show()

	plt.title('Max Q Values vs episodes')
	plt.plot(range(0, num_games), max_q_values, '-o', markersize=5)
	plt.ylabel('Max Q values')
	plt.show()


if __name__ == '__main__':

	args = sys.argv[1]
	plot_train(args)