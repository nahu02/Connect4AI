from typing import Tuple
import logging

import numpy as np

from board import Board

logger = logging.getLogger(__name__)
logging.basicConfig(filename='student_player.log', level=logging.DEBUG)


def heuristic_score(board: Board) -> int:
    """
    Heuristic score of the board state for the given player.
    Works only for 2 players.
    :param board: board state
    :return: number, where higher is better for the given player
    """

    player_index = flip_player_index(board.get_last_player_index())

    if board.game_ended():
        logger.debug('game ended')
        winner = board.get_winner()
        if winner == 0:
            return 0
        if winner == player_index:
            return np.inf
        else:
            return -np.inf

    score = 0
    """
    getting all exposed pieces:
        everything that has a 0 as a neighbour
    
    for all exposed pieces:
        can I make it longer?
            if so, how many connected to it already?
            apply weights and add/subtract to score
        can I make it taller?
            if so, how many connected to it already?
            apply weights and add/subtract to score
        can I expand it diagonally to the right (skew)?
            if so, how many connected to it already?
            apply weights and add/subtract to score
        can I expand it diagonally to the left?
            if so, how many connected to it already?
            apply weights and add/subtract to score
    """
    exponential_weight = 2

    on_the_right_edge = lambda a, b: a == board.get_state().shape[1] - 1
    on_the_left_edge = lambda a, b: a == 0
    on_the_top_edge = lambda a, b: b == 0
    on_the_bottom_edge = lambda a, b: b == board.get_state().shape[0] - 1

    exposed_left = lambda a, b: not on_the_left_edge(a, b) and board.get_state()[b][a - 1] == 0
    exposed_right = lambda a, b: not on_the_right_edge(a, b) and board.get_state()[b][a + 1] == 0
    exposed_top = lambda a, b: not on_the_top_edge(a, b) and board.get_state()[b - 1][a] == 0
    exposed_top_right = lambda a, b: not on_the_top_edge(a, b) and not on_the_right_edge(a, b) and board.get_state()[b - 1][a + 1] == 0
    exposed_top_left = lambda a, b: not on_the_top_edge(a, b) and not on_the_left_edge(a, b) and board.get_state()[b - 1][a - 1] == 0
    exposed_bottom_right = lambda a, b: not on_the_bottom_edge(a, b) and not on_the_right_edge(a, b) and board.get_state()[b + 1][a + 1] == 0
    exposed_bottom_left = lambda a, b: not on_the_bottom_edge(a, b) and not on_the_left_edge(a, b) and board.get_state()[b + 1][a - 1] == 0

    change_in_score = lambda n, player: (exponential_weight ** n) * (1 if player == player_index else -1)

    for (y, x), val in np.ndenumerate(board.get_state()):
        if val == 0:
            continue

        # horizontal left
        if exposed_left(x, y):
            if exposed_bottom_left(x, y):  # it cannot be connected to anything (for now)
                length = 0
            else:
                length = 1
            n_x = x  # neighbour x
            while not on_the_right_edge(n_x, y)\
                    and board.get_state()[y][n_x + 1] == val:
                n_x += 1
                length += 1
            score += change_in_score(length, val)

        # horizontal right
        if exposed_right(x, y):
            if exposed_bottom_right(x, y):
                length = 0
            else:
                length = 1
            n_x = x  # neighbour x
            while not on_the_left_edge(n_x, y)\
                    and board.get_state()[y][n_x - 1] == val:
                n_x -= 1
                length += 1
            score += change_in_score(length, val)

        # vertical
        if exposed_top(x, y):
            length = 1
            n_y = y  # neighbour y
            while not on_the_bottom_edge(x, n_y)\
                    and board.get_state()[n_y - 1][x] == val:
                n_y -= 1
                length += 1
            score += change_in_score(length, val)

        # skew
        if exposed_top_right(x, y):
            if exposed_right(x, y):  # it cannot be connected to anything (for now)
                length = 0
            else:
                length = 1
            n_x = x  # neighbour x
            n_y = y  # neighbour y
            while not on_the_right_edge(n_x, n_y)\
                    and not on_the_bottom_edge(n_x, n_y)\
                    and board.get_state()[n_y - 1][n_x + 1] == val:
                n_x += 1
                n_y -= 1
                length += 1
            score += change_in_score(length, val)

        # anti-skew
        if exposed_top_left(x, y):
            if exposed_left(x, y):  # it cannot be connected to anything (for now)
                length = 0
            else:
                length = 1
            n_x = x  # neighbour x
            n_y = y  # neighbour y
            while not on_the_left_edge(n_x, n_y)\
                    and not on_the_bottom_edge(n_x, n_y)\
                    and board.get_state()[n_y - 1][n_x - 1] == val:
                n_x -= 1
                n_y -= 1
                length += 1
            score += change_in_score(length, val)

    return score


def negamax(depth: int, board: Board) -> [int, int]:
    """
    Negamax variant of the minimax algorithm. Bounded by depth.
    :param depth: current depth
    :param board: current board state
    :return: the best column to play and the score after playing the best column for the given player
    """
    player_index = flip_player_index(board.get_last_player_index())

    if depth == 0 or board.game_ended():
        logging.debug(f"depth: {depth}, score: {heuristic_score(board)} if player {player_index} plays column {board.get_last_player_column()}")
        if board.game_ended():
            logging.debug(f"game ended, winner: {board.get_winner()}, score: {heuristic_score(board)}")
        return board.get_last_player_column(), heuristic_score(board)

    best_score = -np.inf
    best_column = -1

    for possible_step in board.get_valid_steps():
        board_copy = board.copy()
        board_copy.step(player_index, possible_step)
        column, score = negamax(depth - 1, board_copy)
        score = -score  # negamax
        if score > best_score:
            best_column = column
            best_score = score

    return best_column, best_score


def flip_player_index(player_index: int) -> int:
    """
    Flip player index. 1 -> 2, 2 -> 1
    :param player_index: player index
    :return: flipped player index
    """
    return 3 - player_index


class StudentPlayer:
    def __init__(self, player_index: int, board_size: Tuple[int, int], n_to_connect: int):
        self.__n_to_connect = n_to_connect
        self.__board_size = board_size
        self.__player_index = player_index
        self.__other_player_index = flip_player_index(player_index)

        self.__board = Board(self.__board_size, self.__n_to_connect)

    def step(self, last_player_col: int) -> int:
        """
        One step (column selection) of the current player.
        :param last_player_col: [-1, board_size[1]), it is -1 if there was no step yet
        :return: step (column index) of the current player
        """
        if last_player_col != -1:
            self.__board.step(self.__other_player_index, last_player_col)

        col, score = negamax(1, self.__board.copy())
        logging.info(f"Player {self.__player_index} played {col} with score {score}")

        self.__board.step(self.__player_index, col)

        return col
