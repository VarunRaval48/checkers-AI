"""
This file contains various helper methods.
"""


import random
from pathlib import Path
import numpy as np

def flip_coin(prob):
	return random.random() < prob

def open_file(file_name, header=None):
    file = Path(file_name)
    if file.is_file():
        print(file_name, 'File exists: Enter (a)/w:')
        c = input()
        if c == "a" or c == '':
            f = open(file_name, "a")
            return f

    f = open(file_name, "w+")
    if header is not None:
        f.write(header)

    return f

def load_weights(weights_file):
    """
    weights_file: file that contains weights to be stored or loaded

    Returns: None if not to load weights else list of weights
    """

    file = Path(weights_file)
    if file.is_file():
    	print(weights_file, 'File exists: use weights:(y)/n:')
    	c = input()
    	if c == "y" or c == '':
    		weights = np.loadtxt(weights_file, delimiter=',', ndmin=2)
    		if len(weights) != 0:
	    		return weights[-1]

    return None