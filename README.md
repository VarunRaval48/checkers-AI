
## Which Python

This project is written using **Python 3.6.4**.

## HOW TO RUN

For a list of available options, enter `python checkers.py -h`

By default, running `python checkers.py` will run a multiplayer checkers game.


## How to Play against Agent

When asked the following question

*./s_ab_3/first_weights File exists: use weights:(y)/n:*

press enter


To see Alpha Beta agent and Reinforcement learning agent playing games, enter following command:

`python checkers.py -f sl -s ab -z ./s_ab_3/first_weights -l 0`


To play game against SARSA agent, enter following command:

`python checkers.py -f sl -s k -z ./s_ab_3/first_weights -l 0`


Press enter when you see a blinking cursor and no input is asked.
To enter moves when asked, for example to move a piece from position [x1, y1] to [x2, y2], 
in start position enter x1 y1 press enter
in end position enter x2 y2 press enter

When there are multiple attack moves like [x1, y1] to [x2, y2] to [x3, y3],
in start position enter x1 y1 press enter
in end position enter x2 y2 x3 y3 press enter


To play game against alphabeta agent, enter following command:

`python checkers.py -f ab -s k`


## References

Board class for checkers game specified in game.py is adapted from the project of [SamRagusa](https://github.com/SamRagusa/Checkers-Reinforcement-Learning).