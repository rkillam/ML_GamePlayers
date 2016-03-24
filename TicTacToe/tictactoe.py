#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Play tic tac toe

Usage:
    tictactoe.py [-t | --test]

Options:
    -t --test  Run unit tests
"""


import numpy as np
import re
import string


# TODO: Have Board inherit from np.ndarray
class Board(object):
    """A tic tac toe board"""

    def __init__(self, size=3):
        self.size = size
        self.board = np.zeros((size, size))

    @property
    def shape(self):
        """Returns the shape of the tic tac toe board

        @rtype: tuple
        @return: the shape of the tic tac toe board

        >>> Board().shape
        (3, 3)
        >>> Board(size=5).shape
        (5, 5)
        """

        return self.board.shape

    def __setitem__(self, index, val):
        """Marks the board by inserting val at self.board[index[0], index[1]]
        using numpy style indexing

        @type index: tuple of ints
        @param index: The index of the cell where val is to be written such that,
        index[0] is the row index and index[1] is the column index

        @param val: The value to be insert into board[index[0], index[1]]

        >>> board = Board()
        >>> board.board
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.]])
        >>> board[0, 0] = 1
        >>> board.board
        array([[ 1.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.]])
        >>> board[-1, -1] = 2
        >>> board.board
        array([[ 1.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  2.]])
        >>> board[:, 1] = 3
        >>> board.board
        array([[ 1.,  3.,  0.],
               [ 0.,  3.,  0.],
               [ 0.,  3.,  2.]])
        """

        r, c = index
        self.board[r, c] = val

    def __getitem__(self, index):
        """Gets the value of the board at self.board[index[0], index[1]]
        using numpy style indexing

        @type index: tuple of ints
        @param index: The index of the cell where val is to be written such that,
        index[0] is the row index and index[1] is the column index

        >>> board = Board()
        >>> board.board[0, 0] = 1
        >>> board.board[-1, -1] = 2
        >>> board.board[:, 1] = 3
        >>> board[0, 0]
        1.0
        >>> board[1, 0]
        0.0
        >>> board[1, 1]
        3.0
        >>> board[-1, -1]
        2.0
        >>> board[:, 1]
        array([ 3.,  3.,  3.])
        """

        return self.board[index]

    @property
    def is_full(self):
        """Checks to see if there are any empty cells in the board

        @rtype: bool
        @return: True if there are no cells within the board that evaluate to
        False, False otherwise

        >>> board = Board()
        >>> board.is_full
        False
        >>> board.board[0] = 1
        >>> board.is_full
        False
        >>> board.board[:, 0] = 1
        >>> board.is_full
        False
        >>> board.board[:, :] = 1
        >>> board.is_full
        True
        """
        return np.all(self.board)

    @property
    def is_empty(self):
        """Checks to see all of the cells within the board are empty

        @rtype: bool
        @return: True if all of the cells within the board that evaluate to
        False, False otherwise

        >>> board = Board()
        >>> board.is_empty
        True
        >>> board.board[0] = 1
        >>> board.is_empty
        False
        >>> board.board[:, 0] = 1
        >>> board.is_empty
        False
        >>> board.board[:, :] = 1
        >>> board.is_empty
        False
        """
        return np.all(np.zeros(self.board.shape) == self.board)

    @staticmethod
    def is_winning_straight(straight):
        """Determines if the given straight is a winning straight where a
        winning straight is one that has identical, non-False, elements

        @type straight: A numpy array
        @param straight: The straight that is to evaluated

        @rtype: bool
        @return: True if all of the elements are the same and do not evaluate to False

        >>> import numpy as np
        >>> Board.is_winning_straight(np.arange(3))
        False
        >>> Board.is_winning_straight(np.zeros((1, 3)))
        False
        >>> Board.is_winning_straight(np.array([None] * 3))
        False
        >>> Board.is_winning_straight(np.array([False] * 3))
        False
        >>> Board.is_winning_straight(np.array([True] * 3))
        True
        >>> Board.is_winning_straight(np.ones((1, 3)))
        True
        """
        return np.all(straight == straight[0]) and np.all(straight)

    @property
    def is_won(self):
        """Checks to see if there is a winning line within the board, where a
        winning line is one that is defined by is_winning_straight.

        @return: If there is a winning line, then the value in that line is
        returned, otherwise the method returns False.

        >>> board = Board()
        >>> board.is_won
        False
        >>> board.board[0, 0] = 1
        >>> board.is_won
        False
        >>> board.board[:, 0] = 1
        >>> board.is_won
        1.0
        >>> board = Board()
        >>> board.board[:, 1] = 2
        >>> board.is_won
        2.0
        >>> board = Board()
        >>> board.board[0, :] = 1
        >>> board.is_won
        1.0
        >>> board = Board()
        >>> np.fill_diagonal(board.board, -1)
        >>> board.is_won
        -1.0
        """

        won = False

        # Save the transposed array to keep from computing it twice
        board_T = self.board.T

        # Create a list of straight lines that are to be evaluated for win
        # conditions
        straights = np.concatenate((
            self.board,                            # Rows
            board_T,                               # Columns
            np.atleast_2d(self.board.diagonal()),  # Diagonal
            np.atleast_2d(board_T.diagonal()),     # Anti-Diagonal
        ))

        # Check straights for a win
        for straight in straights:
            # Does this straight have all the same, non zero, value
            straight_win = Board.is_winning_straight(straight)

            # If it does, then it is a winning line and we should return the
            # value that it holds (i.e. the winner)
            if straight_win:
                won = straight[0]
                break

        return won


    def __str__(self):
        return str(self.board)


def extract_ints(i_str):
    """Extracts all of the ints contained in the given string, if there are
    consecutive digits they are considered to be the same number

    >>> extract_ints('1, 2')
    [1, 2]
    >>> extract_ints('hello 10, 12')
    [10, 12]
    >>> extract_ints('hello 10,12')
    [10, 12]
    >>> extract_ints('')
    []
    >>> extract_ints('1')
    [1]
    >>> extract_ints('1.0')
    [1, 0]
    """

    # Get all of the numbers
    nums = re.split(r'\D', i_str.strip())

    # Convert them into ints and ignore all empty strings
    return [int(num) for num in nums if num]


def main():
    ttt_board = Board()

    players = (1, 2)
    cur_player_index = 0

    while not ttt_board.is_won:
        try:
            cur_player = players[cur_player_index]

            i_str = input('{}\nPlayer {}\'s turn > '.format(ttt_board, cur_player))
            r, c = extract_ints(i_str)

            ttt_board[r, c] = cur_player

            cur_player_index += 1
            if cur_player_index >= len(players):
                cur_player_index = 0
        except IndexError as e:
            print(e)

    print('Player: {} wins'.format(ttt_board.is_won))


if __name__ == '__main__':
    import docopt
    import doctest

    args = docopt.docopt(__doc__)

    if args['--test']:
        doctest.testmod()

    else:
        main()
