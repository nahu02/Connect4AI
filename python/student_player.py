from typing import Tuple
import logging

import numpy as np

from board import Board

logger = logging.getLogger(__name__)
logging.basicConfig(filename='student_player.log', level=logging.DEBUG)


def how_many_n_in_a_row(n: int, board_state: np.ndarray, player_index: int) -> int:
    """
    Counts how many times there are n in a row for player_index.
    For example, for a row [0, 1, 1, 1, 0, 0, 1, 0, 1, 1] and n = 3, then there are 2 2 in a row for player 1.
    But a sequence of 1, 1, 1, 1 there is only one 2 in a row.
    :param n: how many adjacent cells to check for per row
    :param board_state: state of the board (board.get_state())
    :param player_index: player index to check for
    :return: number of times there are n in a row for player_index
    """
    n_in_a_row = 0
    for row in board_state:
        in_the_row = 0
        for cell in row:
            if cell == player_index:
                in_the_row += 1
            else:
                if in_the_row >= n:
                    n_in_a_row += 1
                in_the_row = 0
    return n_in_a_row


def how_many_n_in_a_column(n: int, board_state: np.ndarray, player_index: int) -> int:
    """
    Counts how many times there are n in a column for player_index.
    :param n: how many adjacent cells to check for per column
    :param board_state: state of the board (board.get_state())
    :param player_index: player index to check for
    :return: number of times there are n in a column for player_index
    """
    return how_many_n_in_a_row(n, board_state.T, player_index)


def how_many_n_anti_skew(n: int, board_state: np.ndarray, player_index: int) -> int:
    """
    Counts how many times there are n in an anti-skew (left to right diagonal) for player_index.
    :param n: how many 'adjacent' cells to check for per anti-skew
    :param board_state: state of the board (board.get_state())
    :param player_index: player index to check for
    :return: number of times there are n in an anti-skew for player_index
    """
    height = board_state.shape[0]
    width = board_state.shape[1]
    anti_skew_lines = [board_state.diagonal(i) for i in range(-height + 1, width)]

    n_in_an_anti_skew = 0
    for line in anti_skew_lines:
        if len(line) < n:
            continue
        in_the_line = 0
        for cell in line:
            if cell == player_index:
                in_the_line += 1
            else:
                if in_the_line >= n:
                    n_in_an_anti_skew += 1
                in_the_line = 0
    return n_in_an_anti_skew


def how_many_n_skew(n: int, board_state: np.ndarray, player_index: int) -> int:
    """
    Counts how many times there are n in a skew (right to left diagonal) for player_index.
    :param n: how many 'adjacent' cells to check for per skew
    :param board_state: state of the board (board.get_state())
    :param player_index:  player index to check for
    :return: number of times there are n in a skew for player_index
    """
    return how_many_n_anti_skew(n, np.flip(board_state, axis=1), player_index)


def heuristic_score(board: Board) -> int:
    """
    Heuristic score of the board state for the player that just played.
    Works only for 2 players, and for 4 in a row.
    :param board: board state
    :return: number, where higher is better for the given player
    """

    player_index = flip_player_index(board.get_last_player_index())

    if board.game_ended():
        logger.debug("game ended")
        winner = board.get_winner()
        if winner == 0:
            return 0
        if winner == player_index:
            return np.inf
        else:
            return -np.inf

    if board.get_state().shape == (6, 7):
        return alternate_heuristic_score_for_6_by_7_board(board)

    weighed_plus = \
        2 * how_many_n_in_a_row(2, board.get_state(), player_index) + \
        2 * how_many_n_in_a_column(2, board.get_state(), player_index) + \
        2 * how_many_n_anti_skew(2, board.get_state(), player_index) + \
        2 * how_many_n_skew(2, board.get_state(), player_index) + \
        4 * how_many_n_in_a_row(3, board.get_state(), player_index) + \
        4 * how_many_n_in_a_column(3, board.get_state(), player_index) + \
        4 * how_many_n_anti_skew(3, board.get_state(), player_index) + \
        4 * how_many_n_skew(3, board.get_state(), player_index)

    weighed_minus = \
        2 * how_many_n_in_a_row(2, board.get_state(), flip_player_index(player_index)) + \
        2 * how_many_n_in_a_column(2, board.get_state(), flip_player_index(player_index)) + \
        2 * how_many_n_anti_skew(2, board.get_state(), flip_player_index(player_index)) + \
        2 * how_many_n_skew(2, board.get_state(), flip_player_index(player_index)) + \
        4 * how_many_n_in_a_row(3, board.get_state(), flip_player_index(player_index)) + \
        4 * how_many_n_in_a_column(3, board.get_state(), flip_player_index(player_index)) + \
        4 * how_many_n_anti_skew(3, board.get_state(), flip_player_index(player_index)) + \
        4 * how_many_n_skew(3, board.get_state(), flip_player_index(player_index))

    return weighed_plus - weighed_minus


def alternate_heuristic_score_for_6_by_7_board(board: Board) -> int:
    """
    Heuristic score of the board state for the player that just played.
    Does not check for game end, assumes this has already been checked.
    Based on http://connect4hci.weebly.com/.
    :param board: board state
    :return: number, where higher is better for the given player
    """
    player_index = flip_player_index(board.get_last_player_index())

    # map of how "good" each cell is, based on how many of all possible 4 in a rows it is a part of
    # I named it allis_map because it's based on the Allis paper, that solved connect 4
    allis_map = np.array([
        [3, 4,  5,  7,  5, 4, 3],
        [4, 6,  8,  9,  8, 6, 4],
        [5, 8, 11, 13, 11, 8, 5],
        [5, 8, 11, 13, 11, 8, 5],
        [4, 6,  8,  9,  8, 6, 4],
        [3, 4,  5,  7,  5, 4, 3]])

    score = 0
    for (r, c), val in np.ndenumerate(board.get_state()):
        if val == player_index:
            score += allis_map[r][c]
        elif val == flip_player_index(player_index):
            score -= allis_map[r][c]

    return score


def negamax(depth: int, board: Board, alpha: int = -np.inf, beta: int = np.inf) -> [int, int]:
    """
    Negamax variant of the minimax algorithm. Bounded by depth.
    :param depth: current depth
    :param board: current board state
    :param alpha: alpha value for alpha-beta pruning
    :param beta: beta value for alpha-beta pruning
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

    for possible_step in board.get_valid_steps():  # might be worth ordering the steps
        board_copy = board.copy()
        board_copy.step(player_index, possible_step)
        column, score = negamax(depth - 1, board_copy, -beta, -alpha)
        score = -score  # negamax
        if score > best_score:
            best_column = column
            best_score = score
        alpha = max(alpha, score)
        if alpha >= beta:
            break

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

        col, score = negamax(4, self.__board.copy())
        logging.info(f"Player {self.__player_index} played {col} with score {score}")

        self.__board.step(self.__player_index, col)

        return col
