#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Play tic tac toe

Usage:
    tictactoe.py [-t | --test]

Options:
    -t --test  Run unit tests
"""


import abc
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


    def __str__(self):
        return str(self.board)


class AbstractPlayer(object, metaclass=abc.ABCMeta):
    """An abstract player class"""

    num_players = 0

    def __init__(self):
        AbstractPlayer.num_players += 1
        self.name = AbstractPlayer.num_players

    @abc.abstractmethod
    def make_move(self, board):
        """Decides what move to make given the board

        @type board: numpy.ndarray
        @param board: The tic tac toe board as it currently stands

        @rtype: tuple of ints
        @return: The cell where the human player wants to put their move
        """

    def __repr__(self):
        return str(self.name)

    def __str__(self):
        return str(self.name)


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


def main():
    # Create the Tic Tac Toe board
    board = Board()

    # Create the list of players
    players = (HumanPlayer(), HumanPlayer())
    cur_player_index = 0

    while not board.is_won:
        try:
            # Get the current player
            cur_player = players[cur_player_index]

            # Get the player's move choice
            r, c = cur_player.make_move(board)

            # Make the player's move
            board[r, c] = cur_player.name

            # Change to the next player
            cur_player_index += 1
            if cur_player_index >= len(players):
                cur_player_index = 0

        except (IndexError, InvalidMove) as e:
            print(e)

        except ValueError:
            print('Invalid input')

    print('Player: {} wins'.format(board.is_won))


if __name__ == '__main__':
    import docopt
    import doctest

    args = docopt.docopt(__doc__)

    if args['--test']:
        doctest.testmod()

    else:
        main()
