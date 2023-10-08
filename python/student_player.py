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


def how_many_options_for_winning_in_next_move(board: Board, player_index: int) -> int:
    """
    Looks at all the legal moves player_index can make and counts how many of them will result in a win.
    :param board: board state
    :param player_index: player index to check all the legal moves for
    :return: number between 0 and board width (inclusive) that represents how many legal moves will result in a win
    """
    result = 0
    for move in board.get_valid_steps():
        board_copy = board.copy()
        board_copy.step(player_index, move)
        if board_copy.game_ended() and board_copy.get_winner() == player_index:
            result += 1
    return result


def heuristic_score(board: Board) -> int:
    """
    Heuristic score of the board.
    Works only for 2 players, with ids 1 and 2. 2 is maximizing, 1 is minimizing.
    :param board: board state
    :return: number, where player id 2 is maximizing and player id 1 is minimizing
    """

    maximizing = 2
    minimizing = 1

    # if the game ended, return the score based on who won
    if board.game_ended():
        if board.get_winner() == maximizing:
            return 1000
        elif board.get_winner() == minimizing:
            return -1000
        else:
            return 0

    # 3 in a row +10
    connecting_threes_score = 10 * (how_many_n_in_a_row(3, board.get_state(), maximizing) +
                                    how_many_n_in_a_column(3, board.get_state(), maximizing) +
                                    how_many_n_anti_skew(3, board.get_state(), maximizing) +
                                    how_many_n_skew(3, board.get_state(), maximizing))

    # opponent's 2 in a row -4
    o_connecting_twos_score = 4 * (how_many_n_in_a_row(2, board.get_state(), minimizing) +
                                   how_many_n_in_a_column(2, board.get_state(), minimizing) +
                                   how_many_n_anti_skew(2, board.get_state(), minimizing) +
                                   how_many_n_skew(2, board.get_state(), minimizing))

    # opponent's winnable 3 in a rows -120
    o_winnables_score = 120 * how_many_options_for_winning_in_next_move(board, minimizing)

    base_score = get_allis_score(board)

    return base_score + connecting_threes_score - o_connecting_twos_score - o_winnables_score


def get_allis_score(board: Board) -> int:
    """
    A base heuristic function for the board state.
    Inspired by http://connect4hci.weebly.com/, that is based on the Allis paper, that solved connect 4
    :param board: board state
    :return:
    """
    maximizing = 2
    minimizing = 1
    a_map = np.array([
        [3, 4,  5,  7,  5, 4, 3],
        [4, 6,  8,  9,  8, 6, 4],
        [5, 8, 11, 13, 11, 8, 5],
        [5, 8, 11, 13, 11, 8, 5],
        [4, 6,  8,  9,  8, 6, 4],
        [3, 4,  5,  7,  5, 4, 3]])

    allis_score = 0
    for (r, c), val in np.ndenumerate(board.get_state()):
        if val == maximizing:
            allis_score += a_map[r][c]
        elif val == minimizing:
            allis_score -= a_map[r][c]

    return allis_score

def minimax(depth: int, board: Board, maximizing: bool, alpha: int = -np.inf, beta: int = np.inf) -> [int, int]:
    """
    Minimax algorithm. Bounded by depth.
    :param depth: current depth
    :param board: current board state
    :param maximizing: whether the current player is maximizing or not. AI should be maximizing, opponent is minimizing.
    :param alpha: alpha value for alpha-beta pruning
    :param beta: beta value for alpha-beta pruning
    :return: the score after playing the best column for the given player, and that column
    """
    maximizing_player_index = 2
    minimizing_player_index = 1

    if depth == 0 or board.game_ended():
        return heuristic_score(board), -1

    if maximizing:
        best_score = -np.inf
        best_col = -1
        for col in board.get_valid_steps():
            board_copy = board.copy()
            board_copy.step(maximizing_player_index, col)
            score, _ = minimax(depth - 1, board_copy, False, alpha, beta)
            logger.debug(f"for col {col}, score is {score}")
            if score > best_score:
                best_score = score
                best_col = col
            if best_score > beta:
                logger.debug(f"pruning with beta, score: {score}, beta: {beta}")
                break
            alpha = max(alpha, best_score)
    else:
        best_score = np.inf
        best_col = -1
        for col in board.get_valid_steps():
            board_copy = board.copy()
            board_copy.step(minimizing_player_index, col)
            score, _ = minimax(depth - 1, board_copy, True, alpha, beta)
            if score < best_score:
                best_score = score
                best_col = col
            if best_score < alpha:
                logger.debug(f"pruning with alpha, score: {score}, alpha: {alpha}")
                break
            beta = min(beta, best_score)

    if best_col == -1:
        logger.warning("No valid step found in minimax")
        best_col = board.get_valid_steps()[0]

    logger.debug(f"Minimax: depth: {depth}, maximizing: {maximizing}, best_col: {best_col}, best_score: {best_score}")

    return best_score, best_col


class StudentPlayer:
    def __init__(self, player_index: int, board_size: Tuple[int, int], n_to_connect: int):
        self.__n_to_connect = n_to_connect
        self.__board_size = board_size
        self.__player_index = player_index
        self.__other_player_index = 3 - player_index

        self.__board = Board(self.__board_size, self.__n_to_connect)

    def step(self, last_player_col: int) -> int:
        """
        One step (column selection) of the current player.
        :param last_player_col: [-1, board_size[1]), it is -1 if there was no step yet
        :return: step (column index) of the current player
        """
        if last_player_col != -1:
            self.__board.step(self.__other_player_index, last_player_col)

        score, col = minimax(6, self.__board.copy(), True)
        logging.info(f"Player {self.__player_index} played {col} with score {score}")

        self.__board.step(self.__player_index, col)

        return col
