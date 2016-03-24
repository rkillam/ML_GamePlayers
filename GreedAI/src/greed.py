#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Main greed file, holds the code for basic gameplay and a base human player
class

Usage:
    greed.py [--player=<PLAYER_TYPE>] [--config=<CONFIG_FILE>]

Options:
    --player=<PLAYER_TYPE>      Specifies what type of player will play the game. Options: [{PLAYER_TYPES}] [default: human]
    --config=<CONFIG_FILE>      The path to the configuration file for the game [default: greed.conf]
"""


import configparser
import random

import players


class GreedBoard(object):
    """A class to hold the greed board"""

    def __init__(self, *, height=None, width=None, player=None):
        self.height = height
        self.width = width
        self.player = player

        self.board = None

        self._init_board()

    @classmethod
    def from_config(cls, *, config=None, **kwargs):
        """Creates a GreedBoard using the configuration, config"""

        return cls(
            height=int(config['height']),
            width=int(config['width']),
            **kwargs
        )

    @property
    def dimensions(self):
        return (self.height, self.width)

    def as_input_vector(self):
        """Converts self.board into a 1D vector where None becomes -1 and the
        player becomes 0
        """

        vector = []
        for row in self.board:
            for cell in row:
                if cell is None:
                    cell = -1

                elif cell is self.player:
                    cell = 0

                vector.append(cell)

        return vector

    def _gen_board(self):
        """Generates a (self.height X self.width) board with random numbers
        from 1 to 9
        """

        self.board = [
            [random.randint(1, 9) for _ in range(self.width)]
            for _ in range(self.height)
        ]

    def _place_player(self):
        """Places the self.player in a random square in the board"""

        self.player.row = random.randint(0, self.height - 1)
        self.player.col = random.randint(0, self.width - 1)

        self.board[self.player.row][self.player.col] = self.player

    def _init_board(self):
        """Generates a game board and places the self.player in a random"""

        self._gen_board()
        self._place_player()

    def in_bounds(self, r, c):
        """Determines if the coordinates (r, c) are within the bounds of the
        board
        """

        return 0 <= r < len(self.board) and 0 <= c < len(self.board[0])

    def get_valid_moves(self):
        """Returns all moves that the player can make that will leave the
        player in the grid
        """

        moves = []
        p_row, p_col = self.player.row, self.player.col

        for control in self.player.controls.values():
            try:
                final_row, final_col = self.get_new_player_pos(control)

                row_vals = self.get_seq_vals(p_row, final_row)
                col_vals = self.get_seq_vals(p_col, final_col)

                if self.in_bounds(final_row, final_col) and \
                    all(self.board[r][c] is not None for r, c in zip(row_vals, col_vals)):

                    moves.append(control)

            except (IndexError, TypeError):
                pass

        return moves

    def get_new_player_pos(self, control):
        """Get the final position of the player given the control direction"""

        # Get the player's current position
        p_row, p_col = self.player.row, self.player.col

        # Get the cell that holds the number of square to move (first in the
        # given direction)
        dir_row, dir_col = p_row + control[0], p_col + control[1]

        # Get the final row and column values of the player's position
        final_row = p_row + (self.board[dir_row][dir_col] * control[0])
        final_col = p_col + (self.board[dir_row][dir_col] * control[1])

        return final_row, final_col

    def get_seq_vals(self, start_pos, end_pos):
        """Get an iterable of the values between start_pos and end_pos. If
        start_pos == end_pos then a infinite generator is given that will yield
        start_pos
        """

        def infinite_generator(val):
            """Continuously yields the given value"""

            while True:
                yield val

        if start_pos != end_pos:
            diff = 1 if end_pos > start_pos else -1
            vals = range(start_pos, end_pos, diff)

        else:
            vals = infinite_generator(start_pos)

        return vals

    def move_player(self, player_move):
        """Moves the player in the direction of player_move the appropriate
        number of spaces, and manages the board as the player travels
        """

        cp_row, cp_col = self.player.row, self.player.col
        final_row, final_col = self.get_new_player_pos(player_move)

        row_vals = self.get_seq_vals(cp_row, final_row)
        col_vals = self.get_seq_vals(cp_col, final_col)

        for r, c in zip(row_vals, col_vals):
            self.board[r][c] = None
            self.player.num_squares_eaten += 1

        # Mark the player down at their new location
        self.board[final_row][final_col] = self.player
        self.player.row, self.player.col = final_row, final_col

    def __str__(self):
        row_strs = [''.join(str(d) if d is not None else ' ' for d in row) for row in self.board]

        return '\n'.join(row_strs)


def main(args):
    config = configparser.ConfigParser()
    config.read(args['--config'])

    board_config = config['board']

    player = players.player_registry[args['--player']]((int(board_config['height']), int(board_config['width'])))
    greed_board = GreedBoard.from_config(config=board_config, player=player)

    while greed_board.get_valid_moves():
        print(greed_board)

        player_move = None
        while player_move not in greed_board.get_valid_moves():
            player_move = player.move(greed_board.as_input_vector())

        greed_board.move_player(player_move)
        print('\n')

    print(greed_board)
    print('Player at {} / {} = {:.2f}%'.format(
        player.num_squares_eaten,
        greed_board.height * greed_board.width,
        player.num_squares_eaten / (greed_board.height * greed_board.width) * 100
    ))


if __name__ == '__main__':
    import docopt

    formatted_doc = __doc__.format(
        PLAYER_TYPES=', '.join(players.player_registry.keys)
    )

    args = docopt.docopt(formatted_doc)
    main(args)
