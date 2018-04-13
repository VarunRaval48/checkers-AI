import copy
import sys

from game import *
from agents import *

class GameState:
    """
    A class which stores information about the state of a game.
    This class uses class Board to perform moves and to check whether game is won or lost.
    """


    def __init__(self, prev_state=None, the_player_turn=True):
        """
        prev_state: an instance of GameState or None
        """

        if prev_state is None:
            prev_spots = None
        else:
            prev_spots = copy.deepcopy(prev_state.board.spots)

        self.board = Board(prev_spots, the_player_turn)


    def get_legal_actions(self):
        """
        Returns the legal moves as list of moves. A single move is a list of positions going from
        first position to next position
        """
        return self.board.get_possible_next_moves()


    def generate_successor(self, action, switch_player_turn=True):
        """
        action is a list of positions indicating move from position at first index to position at
        next index

        Returns: a new state without any changes to current state
        """

        successor_state = GameState(self, self.board.player_turn)
        successor_state.board.make_move(action, switch_player_turn)

        return successor_state

    def is_first_agent_turn(self):
        """
        Returns: True if it is the turn of first agent else returns False
        """
        return self.board.player_turn


    def is_game_over(self):
        """
        Returns: True if either agent has won the game
        """
        return self.board.is_game_over()

    def is_first_agent_win(self):
        """
        Returns: False if game is still on or first agent has lost and True iff first agent has won
        """

        if not self.is_game_over() or self.is_first_agent_turn():
            return False

        return True

    def is_second_agent_win(self):
        """
        Returns: False if game is still on or second agent has lost and True iff second agent has won
        """

        if not self.is_game_over() or not self.is_first_agent_turn():
            return False

        return True


    def print_board(self):
        self.board.print_board()


    def player_info(self):
        """
        Returns: the index of player (P1 or P2) whose turn is next
        """

        # if player_turn is true, it indicates turn of player P1
        return self.board.P1 if self.board.player_turn else self.board.P2


    def player_symbol(self, index):
        """
        index: index of the player to be queried 1 or 2

        Returns: symbol corresponding to the player in the game
        """
        if index == 1:
            return self.board.P1_SYMBOL
        else:
            return self.board.P2_SYMBOL


    def get_pieces_and_kings(self, player=True):
        """
        player: True if for the first player, false for the second player

        Returns: the number of pieces and kings for every player in the current state
        """
        spots = self.board.spots
        count = [0,0,0,0]   
        for x in spots:
            for y in x:
                if y != 0:
                    count[y-1] = count[y-1] + 1

        if player:
            return [count[0], count[2]]  #Player 1
        else:
            return [count[1], count[3]]  #Player 2


class ClassicGameRules:
    """
    This class is used to control the flow of game.
    The only control right now is whether to show game board at every step or not.
    """

    def __init__(self, max_moves=200):
        self.max_moves = max_moves
        self.quiet = False

    def new_game(self, first_agent, second_agent, first_agent_turn, quiet=False):
        init_state = GameState(the_player_turn=first_agent_turn)

        self.quiet = quiet
        game = Game(first_agent, second_agent, init_state, self)

        return game


def load_agent(agent_type):
    """
    agent_type: type of agent, e.g. k, ab, rl

    Returns: instance of the respective agent
    """

    if agent_type == 'k':
        return KeyBoardAgent()
    elif agent_type == 'ab':
        return AlphaBetaAgent()
    elif agent_type == 'rl':
        return QLearningAgent()
    else:
        raise Exception('Invalid agent ' + str(agent_type))


def default(str):
    return str + ' [Default: %default]'


def read_command(argv):
    """
    Processes the command used to run pacman from the command line.
    """

    from optparse import OptionParser

    usage_str = """
    USAGE:      python checkers.py <options>
    EXAMPLES:   (1) python checkers.py
                    - starts a two player game
    """
    parser = OptionParser(usage_str)

    parser.add_option('-n', '--numGames', dest='num_games', type='int',
                      help=default('the number of GAMES to play'), metavar='GAMES', default=1)

    # k for keyboard agent
    # ab for alphabeta agent
    # rl for reinforcement learning agent
    parser.add_option('-f', '--agentFirstType', dest='first_agent', type='string',
                      help=default('the first agent of game'), default='k')
    parser.add_option('-s', '--agentSecondType', dest='second_agent', type='string',
                      help=default('the second agent of game'), default='k')

    parser.add_option('-t', '--turn', dest='turn', type='int', 
                      help=default('which agent should take first turn'), default=1)

    parser.add_option('-r', '--numTraining', dest='num_train', type='int',
                      help=default('number of training steps'), default=0)

    options, garbage = parser.parse_args(argv)

    if len(garbage) > 0:
        raise Exception('Command line input not understood ' + str(garbage))

    args = dict()

    args['num_games'] = options.num_games

    args['first_agent'] = load_agent(options.first_agent)

    args['second_agent'] = load_agent(options.second_agent)

    args['first_agent_turn'] = options.turn == 1

    args['num_training'] = options.num_train

    return args



def run_games(first_agent, second_agent, first_agent_turn, num_games, num_training=0):
    """
    first_agent: instance of Agent which reflects first agent
    second_agent: instance of Agent which reflects second agent
    first_agent_turn: True if turn is of the first agent
    num_games: total number of games to run without training
    num_training: total number of training games to run
    """

    for i in range(num_games):
        rules = ClassicGameRules()

        if first_agent.is_learning_agent:
            first_agent.start_learning()

        if second_agent.is_learning_agent:
            second_agent.start_learning()

        game = rules.new_game(first_agent, second_agent, first_agent_turn)

        game.run()


if __name__ == '__main__':
    
    # game_state = GameState()
    # game_state.print_board()

    # # get legal moves from this state with respect to the player whose turn is there
    # moves = game_state.get_legal_actions()
    # print(moves)

    # game_state = game_state.generate_successor([[2,0], [3,0]])
    # game_state.print_board()

    # moves = game_state.get_legal_actions()
    # print(moves)

    # game_state = game_state.generate_successor([[5,1], [4,1]])
    # game_state.print_board()

    # moves = game_state.get_legal_actions()
    # print(moves)

    # game_state = game_state.generate_successor([[3,0], [5,1]])
    # game_state.print_board()

    # moves = game_state.get_legal_actions()
    # print(moves)

    # print(game_state.player_info())


    args = read_command(sys.argv[1:])
    run_games(**args)