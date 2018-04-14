
import random
from pathlib import Path

def flip_coin(prob):
	return random.random() < prob

def open_file(file_name, header=None):
    file = Path(file_name)
    if file.is_file():
        print(file_name, 'File exists: Enter (a)/w:')
        c = input()
        if c == "a":
            f = open(first_file_name, "a")
            return f

    f = open(file_name, "w+")
    if header is not None:
        f.write(header)

    return f
