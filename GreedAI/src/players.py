#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Holds the classes for different greed game players"""

import abc
import numpy as np
import random

import utils


player_registry = utils.Registry(name='Player Registry')


class AbstractPlayer(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        self.row = None
        self.col = None

        self.num_squares_eaten = 0

    @abc.abstractmethod
    def move(self, *args, **kwargs):
        """Returns the direction that the player will move in the form of an
        (y, x) 2-tuple:
            (1, 0)   # Down
            (-1, 0)  # Up
            (0, 1)   # Right
            (0, -1)  # Left
            (1, 1)   # Down-Right
            (1, -1)  # Down-Left
            (-1, 1)  # Up-Right
            (-1, -1) # Up-Left
        """

    def __str__(self):
        return '@'


@player_registry.register('human')
class HumanPlayer(AbstractPlayer):
    """Allows for a human to play the game by using the numpad to select the
    player's direction
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.controls = {
            2: (1, 0),   # Down
            8: (-1, 0),  # Up
            6: (0, 1),   # Right
            4: (0, -1),  # Left
            3: (1, 1),   # Down-Right
            1: (1, -1),  # Down-Left
            9: (-1, 1),  # Up-Right
            7: (-1, -1), # Up-Left
        }

    def move(self, *args, **kwargs):
        return self.controls[int(input('Move > '))]


@player_registry.register('random')
class RandomPlayer(HumanPlayer):
    """Moves in random directions"""

    def move(self, *args, **kwargs):
        return random.choice(list(self.controls.values()))


@player_registry.register('neural_net')
class NeuralNet(RandomPlayer):
    """Given a board will use a neural network to decide in what direction to
    move
    """

    def __init__(self, board_dimensions):
        super().__init__(board_dimensions)

        num_cells = 1
        for dim in board_dimensions:
            num_cells *= dim

        # Create neural network
        self.hidden_layer = np.random.rand(num_cells, len(board_dimensions))

    def _nn_move(self, board):
        """Uses the neural network to generate 2 output values between -1 and 1"""

        return np.dot(board, self.hidden_layer)

    def move(self, board):
        """Uses the neural network to determine what move should be made and
        then formats it such that the GreedBoard can use it
        """

        nn_output = self._nn_move(board)
        print(nn_output)
        return super().move()
