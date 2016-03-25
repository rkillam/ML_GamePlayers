#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Play tic tac toe

Usage:
    tictactoe.py [-t | --test] [-v | --verbose] [--size=<SIZE>]

Options:
    -t --test       Run unit tests
    -v --verbose    Print all output
    --size=<SIZE>   The size of the tic tac toe board [default: 3]
"""


import abc
import numpy as np
import re
import string

import neural_network as nn


# TODO: Have Board inherit from np.ndarray
class Board(object):
    """A tic tac toe board"""

    def __init__(self, size=3):
        self.size = size
        self.board = np.zeros((size, size))

    @property
    def num_cells(self):
        """Returns the number of cells in the tic tac toe board

        @rtype: int
        @return: The number of cells in the tic tac toe board

        >>> Board().num_cells
        9
        >>> Board(size=5).num_cells
        25
        """

        return self.size * self.size

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
        Traceback (most recent call last):
            ...
        InvalidMove: Type Board does not support (slice(None, None, None), 1) as an index
        >>> board[0, 0] = 2
        Traceback (most recent call last):
            ...
        InvalidMove: Cell (0, 0) already taken
        """

        try:
            if not bool(self.board[index]):
                self.board[index] = val
            else:
                raise InvalidMove('Cell {} already taken'.format(index))
        except ValueError:
            raise InvalidMove('Type {} does not support {} as an index'.format(
                self.__class__.__name__,
                index
            ))

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

    def get_random_empty_cell(self):
        """Returns a random cell that has not been claimed by a player
        """

        if self.is_full:
            raise Exception('There are no empty cells')

        # Get the indices of all of the empty cells as a tuple of ndarrays for
        # each dimension
        options = np.where(self.board == 0)

        # Stack the index components and then transpose the matrix so that the
        # indices are held in rows
        options = np.vstack(options).T

        # Randomly choose an empty cell
        choice = np.random.randint(0, options.shape[0])

        return options[choice]


    def __str__(self):
        return str(self.board)


class AbstractPlayer(object, metaclass=abc.ABCMeta):
    """An abstract player class"""

    num_players = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = kwargs.get('name', 'Player')

        AbstractPlayer.num_players += 1
        self.num = AbstractPlayer.num_players

    @abc.abstractmethod
    def make_move(self, board):
        """Decides what move to make given the board

        @type board: numpy.ndarray
        @param board: The tic tac toe board as it currently stands

        @rtype: tuple of ints
        @return: The cell where the human player wants to put their move
        """

    def __repr__(self):
        return str(self.num)

    def __str__(self):
        return str(self.num)


class InvalidMove(Exception):
    """An exception that is caused by a player attempting to make an invalid
    move
    """


class HumanPlayer(AbstractPlayer):
    """A human player who enters commands through the console"""

    @staticmethod
    def extract_ints(i_str):
        """Extracts all of the ints contained in the given string, if there are
        consecutive digits they are considered to be the same number

        @type i_str: str
        @param i_str: A string from which the ints are to be extracted

        @rtype: tuple
        @return: A tuple of ints

        >>> HumanPlayer.extract_ints('1, 2')
        (1, 2)
        >>> HumanPlayer.extract_ints('hello 10, 12')
        (10, 12)
        >>> HumanPlayer.extract_ints('hello 10,12')
        (10, 12)
        >>> HumanPlayer.extract_ints('')
        ()
        >>> HumanPlayer.extract_ints('1')
        (1,)
        >>> HumanPlayer.extract_ints('1.0')
        (1, 0)
        """

        # Get all of the numbers
        nums = re.split(r'\D', i_str.strip())

        # Convert them into ints and ignore all empty strings
        return tuple(int(num) for num in nums if num)

    def make_move(self, board):
        """Read's the human player's move from stdin, extracts the cell
        coordinates as ints and returns them as a tuple.

        @type board: numpy.ndarray
        @param board: The tic tac toe board as it currently stands

        @rtype: tuple of ints
        @return: The cell where the human player wants to put their move
        """

        i_str = input('{}\nPlayer {}\'s turn > '.format(board, self))
        return HumanPlayer.extract_ints(i_str)


class NNPlayer(AbstractPlayer, nn.NeuralNetwork):
    """A player that makes its decisions based on a neural network"""

    def make_move(self, board):
        """Checks to see if there is a winning move, if so then it takes it,
        finally it will take a random cell
        """

        outputs = super().feed_forward(board.board.flatten())

        # Get the row and column indices from the largest cell number
        cell_num = outputs[-1].argmax()
        r = cell_num // board.size
        c = cell_num % board.size

        return r, c

def main(kwargs):
    # Create the Tic Tac Toe board
    board = Board(int(kwargs['--size']))
    nn_dims = (board.num_cells, board.num_cells*2, board.num_cells)

    # Create the list of players
    players = (NNPlayer(dims=nn_dims), NNPlayer(dims=nn_dims))
    cur_player_index = 0

    while not board.is_won and not board.is_full:
        try:
            # Get the current player
            cur_player = players[cur_player_index]

            # Get the player's move choice
            r, c = cur_player.make_move(board)

            # Make the player's move
            board[r, c] = cur_player.num

        except (ValueError, IndexError, InvalidMove) as e:
            print(e)

            r, c = board.get_random_empty_cell()
            board[r, c] = cur_player.num

            print('Making random move to ({}, {})'.format(r, c))

        finally:
            # Change to the next player
            cur_player_index += 1
            if cur_player_index >= len(players):
                cur_player_index = 0

    print('\n\n{}\nPlayer: {} wins'.format(board, board.is_won))


if __name__ == '__main__':
    import docopt
    import doctest

    args = docopt.docopt(__doc__)

    if args['--test']:
        doctest.testmod(verbose=args['--verbose'])

    else:
        main(args)
