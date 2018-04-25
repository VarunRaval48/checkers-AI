"""
This file contains implementation of checkers game.
This file also contains feature, reward functions and
methods to run a single game.
"""

import math
import copy
from functools import reduce


CHECKERS_FEATURE_COUNT = 8
WIN_REWARD = 500
LOSE_REWARD = -500
LIVING_REWARD = -0.1


class Board:

    """
    A class to represent and play an 8x8 game of checkers.
    """
    EMPTY_SPOT = 0
    P1 = 1
    P2 = 2
    P1_K = 3
    P2_K = 4
    BACKWARDS_PLAYER = P2
    HEIGHT = 8
    WIDTH = 4

    P1_SYMBOL = 'o'
    P1_K_SYMBOL = 'O'
    P2_SYMBOL = 'x'
    P2_K_SYMBOL = 'X'


    def __init__(self, old_spots=None, the_player_turn=True):
        """
        Initializes a new instance of the Board class.  Unless specified otherwise, 
        the board will be created with a start board configuration.

        the_player_turn=True indicates turn of player P1

        NOTE:
        Maybe have default parameter so board is 8x8 by default but nxn if wanted.
        """
        self.player_turn = the_player_turn
        if old_spots is None:
            self.spots = [[j, j, j, j] for j in [self.P1, self.P1, self.P1, self.EMPTY_SPOT, 
                                                self.EMPTY_SPOT, self.P2, self.P2, self.P2]]
        else:
            self.spots = old_spots


    def reset_board(self):
        """
        Resets the current configuration of the game board to the original 
        starting position.
        """
        self.spots = Board().spots


    def empty_board(self):
        """
        Removes any pieces currently on the board and leaves the board with nothing but empty spots.
        """
        # TODO Make sure [self.EMPTY_SPOT]*self.HEIGHT] has no issues
        self.spots = [[j, j, j, j] for j in [self.EMPTY_SPOT] * self.HEIGHT]   

    
    def is_game_over(self):
        """
        Finds out and returns weather the game currently being played is over or
        not.
        """
        if not self.get_possible_next_moves():
            return True

        return False


    def not_spot(self, loc):
        """
        Finds out of the spot at the given location is an actual spot on the game board.
        """
        if len(loc) == 0 or loc[0] < 0 or loc[0] > self.HEIGHT - 1 or loc[1] < 0 or \
            loc[1] > self.WIDTH - 1:
            return True
        return False


    def get_spot_info(self, loc):
        """
        Gets the information about the spot at the given location.
        
        NOTE:
        Might want to not use this for the sake of computational time.
        """
        return self.spots[loc[0]][loc[1]]


    def forward_n_locations(self, start_loc, n, backwards=False):
        """
        Gets the locations possible for moving a piece from a given location diagonally
        forward (or backwards if wanted) a given number of times(without directional change midway).
        """
        if n % 2 == 0:
            temp1 = 0
            temp2 = 0
        elif start_loc[0] % 2 == 0:
            temp1 = 0
            temp2 = 1 
        else:
            temp1 = 1
            temp2 = 0

        answer = [[start_loc[0], start_loc[1] + math.floor(n / 2) + temp1], 
                    [start_loc[0], start_loc[1] - math.floor(n / 2) - temp2]]

        if backwards: 
            answer[0][0] = answer[0][0] - n
            answer[1][0] = answer[1][0] - n
        else:
            answer[0][0] = answer[0][0] + n
            answer[1][0] = answer[1][0] + n

        if self.not_spot(answer[0]):
            answer[0] = []
        if self.not_spot(answer[1]):
            answer[1] = []

        return answer


    def get_simple_moves(self, start_loc):
        """
        Gets the possible moves a piece can make given that it does not capture any 
        opponents pieces.

        PRE-CONDITION:
        -start_loc is a location with a players piece
        """
        if self.spots[start_loc[0]][start_loc[1]] > 2:
            next_locations = self.forward_n_locations(start_loc, 1)
            next_locations.extend(self.forward_n_locations(start_loc, 1, True))
        elif self.spots[start_loc[0]][start_loc[1]] == self.BACKWARDS_PLAYER:
            next_locations = self.forward_n_locations(start_loc, 1, True)  
        else:
            next_locations = self.forward_n_locations(start_loc, 1)
        

        possible_next_locations = []

        for location in next_locations:
            if len(location) != 0:
                if self.spots[location[0]][location[1]] == self.EMPTY_SPOT:
                    possible_next_locations.append(location)
            
        return [[start_loc, end_spot] for end_spot in possible_next_locations]


    def get_capture_moves(self, start_loc, move_beginnings=None):
        """
        Recursively get all of the possible moves for a piece which involve capturing an 
        opponent's piece.
        """
        if move_beginnings is None:
            move_beginnings = [start_loc]
            
        answer = []
        if self.spots[start_loc[0]][start_loc[1]] > 2:  
            next1 = self.forward_n_locations(start_loc, 1)
            next2 = self.forward_n_locations(start_loc, 2)
            next1.extend(self.forward_n_locations(start_loc, 1, True))
            next2.extend(self.forward_n_locations(start_loc, 2, True))
        elif self.spots[start_loc[0]][start_loc[1]] == self.BACKWARDS_PLAYER:
            next1 = self.forward_n_locations(start_loc, 1, True)
            next2 = self.forward_n_locations(start_loc, 2, True)
        else:
            next1 = self.forward_n_locations(start_loc, 1)
            next2 = self.forward_n_locations(start_loc, 2)
        
        
        for j in range(len(next1)):
            # if both spots exist
            if (not self.not_spot(next2[j])) and (not self.not_spot(next1[j])) : 
                # if next spot is opponent
                if self.get_spot_info(next1[j]) != self.EMPTY_SPOT and \
                    self.get_spot_info(next1[j]) % 2 != self.get_spot_info(start_loc) % 2:  
                    # if next next spot is empty
                    if self.get_spot_info(next2[j]) == self.EMPTY_SPOT:
                        temp_move1 = copy.deepcopy(move_beginnings)
                        temp_move1.append(next2[j])
                        
                        answer_length = len(answer)
                        
                        if self.get_spot_info(start_loc) != self.P1 or \
                            next2[j][0] != self.HEIGHT - 1: 
                            if self.get_spot_info(start_loc) != self.P2 or next2[j][0] != 0: 

                                temp_move2 = [start_loc, next2[j]]
                                
                                temp_board = Board(copy.deepcopy(self.spots), self.player_turn)
                                temp_board.make_move(temp_move2, False)

                                answer.extend(temp_board.get_capture_moves(temp_move2[1], temp_move1))
                                
                        if len(answer) == answer_length:
                            answer.append(temp_move1)
                            
        return answer


    def get_piece_locations(self):
        """
        Gets all the pieces of the current player
        """
        piece_locations = []
        for j in range(self.HEIGHT):
            for i in range(self.WIDTH):
                if (self.player_turn == True and 
                    (self.spots[j][i] == self.P1 or self.spots[j][i] == self.P1_K)) or \
                (self.player_turn == False and 
                    (self.spots[j][i] == self.P2 or self.spots[j][i] == self.P2_K)):
                    piece_locations.append([j, i])  

        return piece_locations        
    
        
    def get_possible_next_moves(self):
        """
        Gets the possible moves that can be made from the current board configuration.
        """

        piece_locations = self.get_piece_locations()

        try:  #Should check to make sure if this try statement is still necessary 
            capture_moves = list(reduce(lambda a, b: a + b, list(map(self.get_capture_moves, piece_locations))))  # CHECK IF OUTER LIST IS NECESSARY

            if len(capture_moves) != 0:
                return capture_moves

            return list(reduce(lambda a, b: a + b, list(map(self.get_simple_moves, piece_locations))))  # CHECK IF OUTER LIST IS NECESSARY
        except TypeError:
            return []
    
    def make_move(self, move, switch_player_turn=True):
        """
        Makes a given move on the board, and (as long as is wanted) switches the indicator for
        which players turn it is.
        """

        if abs(move[0][0] - move[1][0]) == 2:
            for j in range(len(move) - 1):
                if move[j][0] % 2 == 1:
                    if move[j + 1][1] < move[j][1]:
                        middle_y = move[j][1]
                    else:
                        middle_y = move[j + 1][1]
                else:
                    if move[j + 1][1] < move[j][1]:
                        middle_y = move[j + 1][1]
                    else:
                        middle_y = move[j][1]
                        
                self.spots[int((move[j][0] + move[j + 1][0]) / 2)][middle_y] = self.EMPTY_SPOT


        self.spots[move[len(move) - 1][0]][move[len(move) - 1][1]] = self.spots[move[0][0]][move[0][1]]
        if move[len(move) - 1][0] == self.HEIGHT - 1 and self.spots[move[len(move) - 1][0]][move[len(move) - 1][1]] == self.P1:
            self.spots[move[len(move) - 1][0]][move[len(move) - 1][1]] = self.P1_K
        elif move[len(move) - 1][0] == 0 and self.spots[move[len(move) - 1][0]][move[len(move) - 1][1]] == self.P2:
            self.spots[move[len(move) - 1][0]][move[len(move) - 1][1]] = self.P2_K
        else:
            self.spots[move[len(move) - 1][0]][move[len(move) - 1][1]] = self.spots[move[0][0]][move[0][1]]
        self.spots[move[0][0]][move[0][1]] = self.EMPTY_SPOT

        if switch_player_turn:
            self.player_turn = not self.player_turn

    def get_potential_spots_from_moves(self, moves):
        """
        Get's the potential spots for the board if it makes any of the given moves.
        If moves is None then returns it's own current spots.
        """
        if moves is None:
            return self.spots
        answer = []
        for move in moves:
            original_spots = copy.deepcopy(self.spots)
            self.make_move(move, switch_player_turn=False)
            answer.append(self.spots) 
            self.spots = original_spots 
        
        return answer

    def insert_pieces(self, pieces_info):
        """
        Inserts a set of pieces onto a board.
        pieces_info is in the form: [[vert1, horz1, piece1], [vert2, horz2, piece2], ..., [vertn, horzn, piecen]]
        """
        for piece_info in pieces_info:
            self.spots[piece_info[0]][piece_info[1]] = piece_info[2]


    def get_symbol(self, location):
        """
        Gets the symbol for what should be at a board location.
        """
        if self.spots[location[0]][location[1]] == self.EMPTY_SPOT:
            return " "
        elif self.spots[location[0]][location[1]] == self.P1:
            return self.P1_SYMBOL
        elif self.spots[location[0]][location[1]] == self.P2:
            return self.P2_SYMBOL
        elif self.spots[location[0]][location[1]] == self.P1_K:
            return self.P1_K_SYMBOL
        else:
            return self.P2_K_SYMBOL


    def print_board(self):
        """
        Prints a string representation of the current game board.
        """

        index_columns = "   "
        for j in range(self.WIDTH):
            index_columns += " " + str(j) + "   " + str(j) + "  "
        print(index_columns)

        norm_line = "  |---|---|---|---|---|---|---|---|"
        print(norm_line)

        for j in range(self.HEIGHT):
            temp_line = str(j) + " "
            if j % 2 == 1:
                temp_line += "|///|"
            else:
                temp_line += "|"
            for i in range(self.WIDTH):
                temp_line = temp_line + " " + self.get_symbol([j, i]) + " |"
                if i != 3 or j % 2 != 1:  # TODO should figure out if this 3 should be changed to self.WIDTH-1
                    temp_line = temp_line + "///|"
            print(temp_line)
            print(norm_line)



def checkers_features(state, action):
    """
    state: game state of the checkers game
    action: action for which the feature is requested

    Returns: list of feature values for the agent whose turn is in the current state
    """
    next_state = state.generate_successor(action, False)

    agent_ind = 0 if state.is_first_agent_turn() else 1
    oppn_ind = 1 if state.is_first_agent_turn() else 0

    num_pieces_list = state.get_pieces_and_kings()

    agent_pawns = num_pieces_list[agent_ind]
    agent_kings = num_pieces_list[agent_ind + 2]
    agent_pieces = agent_pawns + agent_kings

    oppn_pawns = num_pieces_list[oppn_ind]
    oppn_kings = num_pieces_list[oppn_ind + 2]
    oppn_pieces = oppn_pawns + oppn_kings


    num_pieces_list_n = next_state.get_pieces_and_kings()

    agent_pawns_n = num_pieces_list_n[agent_ind]
    agent_kings_n = num_pieces_list_n[agent_ind + 2]
    agent_pieces_n = agent_pawns_n + agent_kings_n

    oppn_pawns_n = num_pieces_list_n[oppn_ind]
    oppn_kings_n = num_pieces_list_n[oppn_ind + 2]
    oppn_pieces_n = oppn_pawns_n + oppn_kings_n

    features = []

    # features.append(agent_pawns_n - agent_pawns)
    # features.append(agent_kings_n - agent_kings)
    # features.append(agent_pieces_n - agent_pieces)

    # pawns and kings of agent and opponent in current state
    features.append(agent_pawns)
    features.append(agent_kings)
    features.append(oppn_pawns)
    features.append(oppn_kings)

    features.append(oppn_pawns_n - oppn_pawns)
    features.append(oppn_kings_n - oppn_kings)
    features.append(oppn_pieces_n - oppn_pieces)

    features.append(next_state.num_attacks())

    # print(features)
    return features


def checkers_reward(state, action, next_state):

    if next_state.is_game_over():
        # infer turn from current state, because at the end same state is used by both agents
        if state.is_first_agent_turn():
            return WIN_REWARD if next_state.is_first_agent_win() else LOSE_REWARD
        else:
            return WIN_REWARD if next_state.is_second_agent_win() else LOSE_REWARD

    agent_ind = 0 if state.is_first_agent_turn() else 1
    oppn_ind = 1 if state.is_first_agent_turn() else 0

    num_pieces_list = state.get_pieces_and_kings()

    agent_pawns = num_pieces_list[agent_ind]
    agent_kings = num_pieces_list[agent_ind + 2]

    oppn_pawns = num_pieces_list[oppn_ind]
    oppn_kings = num_pieces_list[oppn_ind + 2]

    num_pieces_list_n = next_state.get_pieces_and_kings()

    agent_pawns_n = num_pieces_list_n[agent_ind]
    agent_kings_n = num_pieces_list_n[agent_ind + 2]

    oppn_pawns_n = num_pieces_list_n[oppn_ind]
    oppn_kings_n = num_pieces_list_n[oppn_ind + 2]

    r_1 = agent_pawns - agent_pawns_n
    r_2 = agent_kings - agent_kings_n
    r_3 = oppn_pawns - oppn_pawns_n
    r_4 = oppn_kings - oppn_kings_n

    reward = r_3 * 0.2 + r_4 * 0.3 + r_1 * (-0.4) + r_2 * (-0.5)

    if reward == 0:
        reward = LIVING_REWARD

    return reward


class Game:
    """
    A class to control a game by asking for actions from agents while following game rules.
    """

    def __init__(self, first_agent, second_agent, game_state, rules):
        """
        first_agent: first agent which corresponds to board.player_turn = True
        second_agent: second agent other than first agent
        game_state: state of the game an instance of GameState
        rules: an instance of ClassicGameRules
        """

        self.first_agent = first_agent
        self.second_agent = second_agent
        self.game_state = game_state
        self.rules = rules


    def run(self):

        quiet = self.rules.quiet
        game_state = self.game_state

        learning_agents = []

        if self.first_agent.is_learning_agent:
            learning_agents.append(self.first_agent)

        if self.second_agent.is_learning_agent:
            learning_agents.append(self.second_agent)

        # inform learning agents about new episode start
        for learning_agent in learning_agents:
            learning_agent.start_episode()


        action = None
        num_moves = 0
        while not game_state.is_game_over() and num_moves < self.rules.max_moves:
            # get the agent whose turn is next
            # print('number of pieces', game_state.get_pieces_and_kings(True), game_state.get_pieces_and_kings(False))
            active_agent = self.first_agent if game_state.is_first_agent_turn() else self.second_agent

            if active_agent.is_learning_agent:
                action = active_agent.observation_function(game_state)
            else:
                action = None

            if not quiet:
                game_state.print_board()
                print('Current turn is of agent: ' + str(game_state.player_symbol(game_state.player_info())))
                print('Available moves: ' + str(game_state.get_legal_actions()))
                # game_state.num_attacks()
                input()

            if action is None:
                action = active_agent.get_action(game_state)

            next_game_state = game_state.generate_successor(action)
            self.game_state = next_game_state

            game_state = self.game_state

            num_moves += 1
            # input()

        if num_moves >= self.rules.max_moves:
            game_state.set_max_moves_done()

        # after the game is over, tell learning agents to learn accordingly

        # inform learning agents about new episode end
        for learning_agent in learning_agents:
            learning_agent.observation_function(game_state)
            learning_agent.stop_episode()

        # game_state.print_board()
        # print(num_moves)

        return num_moves, game_state