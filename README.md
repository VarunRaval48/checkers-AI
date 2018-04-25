
## Which Python

This project is written using **Python 3.6.4**.

## HOW TO RUN

For a list of available options, enter `python checkers.py -h`

By default, running `python checkers.py` will run a multiplayer checkers game.


## How to Play against Agent

When asked the question *./s_ab_3/first_weights File exists: use weights:(y)/n:*, press enter


#### See game play between Alpha-Beta and SARSA agent
To see Alpha Beta agent and Reinforcement learning agent playing games, enter following command:

`python checkers.py -f sl -s ab -z ./s_ab_3/first_weights -l 0`


#### Play against SARSA agent
To play game against SARSA agent, enter following command:

`python checkers.py -f sl -s k -z ./s_ab_3/first_weights -l 0`


Press enter when you see a blinking cursor and no input is asked.
To enter moves when asked, for example to move a piece from position [x1, y1] to [x2, y2], 
in start position enter x1 y1 press enter
in end position enter x2 y2 press enter

When there are multiple attack moves like [x1, y1] to [x2, y2] to [x3, y3],
in start position enter x1 y1 press enter
in end position enter x2 y2 x3 y3 press enter


#### Play against Alpha-Beta agent
To play game against alphabeta agent, enter following command:

`python checkers.py -f ab -s k`


## About Alpha-Beta agent

This agent is Minimax agent with Alpha-Beta pruning.

To create this agent, search upto depth 3 is performed, and then evaluation function is used.

Evaluation function is as following:
1. If minimax agent wins, +500
2. If minimax agent loses, −500
3. If none of the above happens, evaluation function is summation of following values:
   * (1) times the number of minimax agent’s pawns
   * (2) times the number of minimax agent’s kings
   * (−1) times the number of opponent’s pawns
   * (−2) times the number of opponent’s kings

## About SARSA agent

Assume that agent is in state *s* and has many choices for action *a*, and for each choice of action, environment takes agent to a state *s'*. For our reinforcement learning agent, we used
following features to represent state action *(s, a)* pairs:
1. Number of agent’s pawns in *s*
2. Number of agent’s kings in *s*
3. Number of opponent’s pawns in *s*
4. Number of opponent’s kings in *s*
5. Difference between number of opponent’s pawns in state *s'* and *s*
6. Difference between number of opponent’s kings in state *s'* and *s*
7. Difference between total number of opponent’s pieces in state *s'* and *s*
8. Number of agent’s pieces being attacked by opponent in state *s'*


#### Reward function
The agent takes action *a* in state *s* and moves to state *s'*. The opponent takes action in state *s'* and
moves to state *s''*. The environment will then given the reward to the agent for the action *a* in state *s*. Thus reward function depends on current state of the agent *s*, the action it took *a* and the state in which it will take next action *s''*.
* Reward function is sum of the following:
  + (−0:4) x (number of agent’s pawns in *s* − number of agent’s pawns in *s''*)
  + (−0:5) x (number of agent’s kings in *s* − number of agent’s kings in *s''*)
  + (+0:2) x (number of opponent’s pawns in *s* − number of opponent’s pawns in *s''*)
  + (+0:3) x (number of opponent’s kings in *s* − number of opponent’s kings in *s''*)
* If agent wins in state *s''* or *s'*, reward of (+500)
* If agent loses in state *s''* or *s'*, reward of (−500)
* If all of above is 0, a living reward of (−0.1)


## Acknowledgements

Board class for checkers game specified in game.py is adapted from the project of [SamRagusa](https://github.com/SamRagusa/Checkers-Reinforcement-Learning).