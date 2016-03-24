#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Miscellaneous functions and classes including logging

Usage:
    utils.py -h | --help
    utils.py -t | --test [-v|--verbose]

Options:
    -h --help       Show this help message
    -t --test       Run the module's unit tests
    -v --verbose    Show all output from the unit tests [optional]
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import collections
import copy
import datetime
import logging
import logging.config
import logging.handlers
import numpy as np
import weakref


NUM_WORKERS = 4


#########################################################################
# Logging

class LevelRangeFileHandler(logging.handlers.RotatingFileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False, low_level=logging.DEBUG, high_level=logging.CRITICAL):
        logging.handlers.RotatingFileHandler.__init__(self, filename, mode, encoding, delay)

        self.low_level = low_level
        self.high_level = high_level

    def emit(self, record):
        if self.low_level <= record.levelno <= self.high_level:
            logging.handlers.RotatingFileHandler.emit(self, record)

# Set configuration file for the logger
logging.config.fileConfig("logging.conf")

# Get our logger as specified in logging.conf
_logger = logging.getLogger("cell")
_log_format = '{obj} - {msg}'


def set_log_level(level):
    _logger.setLevel(level)


def debug(obj, msg):
    """ Logging function """

    _logger.debug(_log_format.format(
        obj=obj,
        msg=msg
    ))


def info(obj, msg):
    """Logging function """

    _logger.info(_log_format.format(
        obj=obj,
        msg=msg
    ))


def warn(obj, msg):
    """Logging function """

    _logger.warn(_log_format.format(
        obj=obj,
        msg=msg
    ))


def error(obj, msg):
    """Logging function """

    _logger.error(_log_format.format(
        obj=obj,
        msg=msg
    ))


def critical(obj, msg):
    """Logging function """

    _logger.critical(_log_format.format(
        obj=obj,
        msg=msg
    ))


def fatal(obj, msg):
    """Logging function """

    _logger.fatal(_log_format.format(
        obj=obj,
        msg=msg
    ))

# End logging
#########################################################################

#########################################################################
# Utility Classes

class CLIMethodManager(object):
    """ A class that is used to register methods for command line arguments.
    That is, if the command has multiple options, this class can be used to
    register those options for later use.
    """

    def __init__(self, name=None):
        self.name = name
        self._methods = {}

    def register(self, name, obj=None):
        """Register the desired object (function/class) using the
        given name.

        @arg name: The name of the object to be registered
        @arg obj [optional]: The object to be registered, if obj is not passed
            then method returns a decorator.
        """

        def helper(obj):
            self._methods[name] = obj
            return obj

        if obj is not None:
            self._methods[name] = obj

        else:
            return helper

    @property
    def methods(self):
        return self._methods.keys()

    @property
    def method_objects(self):
        return self._methods.values()

    def __getitem__(self, method_name):
        return self._methods[method_name]

    def __contains__(self, obj):
        return obj in self._methods

    def __repr__(self):
        return '{CLS_NAME}(name={NAME}, methods={METHODS}'.format(
            CLS_NAME=self.__class__.__name__,
            NAME=self.name,
            METHODS=self.methods
        )


class WeakBoundMethod(object):
    """Used to hold a weak reference to a method, this stops objects from being
    kept simply because one of their methods is registered as a callback
    somewhere
    """

    def __init__(self, method):
        self._self = weakref.ref(method.__self__)
        self._func = method.__func__


class Memento(object):
    """Memento wrapper class that will copy the given object and restore it
    when requested
    """

    def __init__(self, obj, deep=False):
        self._obj = obj
        self._state = copy.deepcopy(obj.__dict__) if deep else copy.copy(obj.__dict__)

    def restore(self):
        self._obj.__dict__.clear()
        self._obj.__dict__.update(self._state)


class OutOfBoundsError(Exception):
    """An exception to denote that an entity is outside of its world's boundaries"""

    def __init__(self, entity, world):
        super().__init__('{} is out of bounds of {}'.format(entity, world))


class Vector(object):
    """Represents a vector object in n dimensional space, holds the logic for
    adding, subtracting vectors and calculating the distance and angle between
    vectors.
    """

    def __init__(self, coordinates):
        super().__init__()
        self.coordinates = np.array(coordinates)

    def distance_between(self, other_vector):
        """Calculates the distance between the self vector and the given vector

        >>> i.distance_between(j)
        1.4142135623730951

        >>> i.distance_between(z)
        1.0

        >>> i.distance_between(i)
        0.0

        >>> v21.distance_between(i)
        1.4142135623730951

        >>> v21.distance_between(j)
        2.0
        """

        return (self - other_vector).magnitude()

    def angle_between(self, other_vector):
        """Calculates the angle between the self vector and the given vector

        >>> i.angle_between(j)
        1.5707963267948966

        >>> i.angle_between(i)
        0.0

        >>> v21.angle_between(i)
        0.46364760900080615
        """

        numerator = np.dot(self.coordinates, other_vector.coordinates)
        denominator = self.magnitude() * other_vector.magnitude()

        return np.arccos(numerator / denominator)

    def magnitude(self):
        """Get the magnitude of the vector

        >>> i.magnitude()
        1.0

        >>> j.magnitude()
        1.0

        >>> z.magnitude()
        0.0

        >>> v21.magnitude()
        2.2360679774997898

        >>> v_1_2.magnitude()
        2.2360679774997898
        """

        return np.sqrt(sum(self.coordinates ** 2))

    def __len__(self):
        """The number of dimensions that the vector exists in

        >>> len(i)
        2
        """

        return len(self.coordinates)

    def __add__(self, other_vector):
        """Vector addition

        >>> (i + j).coordinates
        array([1, 1])

        >>> (i + z).coordinates
        array([1, 0])

        >>> (i + (0, 1)).coordinates
        array([1, 1])

        >>> (i + v21).coordinates
        array([3, 1])

        >>> (i + v_1_2).coordinates
        array([ 0, -2])

        >>> (i + i).coordinates
        array([2, 0])
        """

        try:
            new_pt = self.coordinates + other_vector.coordinates

        except AttributeError:
            new_pt = self.coordinates + np.array(other_vector)

        return self.__class__(new_pt)

    def __radd__(self, other_vector):
        """Ensures that vector addition with a non-vector on the left works

        >>> ((0, 1) + i).coordinates
        array([1, 1])
        """
        return self + other_vector

    def __sub__(self, other_vector):
        """Vector subtraction

        >>> (i - j).coordinates
        array([ 1, -1])

        >>> (j - i).coordinates
        array([-1,  1])

        >>> (i - i).coordinates
        array([0, 0])

        >>> (i - z).coordinates
        array([1, 0])

        >>> (i - (0, -1)).coordinates
        array([1, 1])

        >>> (v21 - v_1_2).coordinates
        array([3, 3])

        >>> (v_1_2 - v21 ).coordinates
        array([-3, -3])
        """

        neg_other_vector = [-coord for coord in other_vector]
        return self + neg_other_vector

    def __rsub__(self, other_vector):
        """Ensures that vector subtraction with a non-vector on the left works

        >>> ((0, 1) - i).coordinates
        array([-1,  1])
        """
        neg_self = self.__class__([-coord for coord in self])
        return neg_self + other_vector

    def __getitem__(self, idx):
        return self.coordinates[idx]

    def __setitem__(self, idx, coord):
        self.coordinates[idx] = coord

    def __iter__(self):
        for coord in self.coordinates:
            yield coord

    def __eq__(self, other_point):
        """Two vectors are equal if all of their coordinates are the same

        >>> Vector((1, 1)) == Vector((1, 1))
        True

        >>> Vector((1, 1)) == Vector((1, 0))
        False

        >>> Vector((1, 0)) == i
        True

        >>> Vector((1, 0)) == j
        False

        >>> Vector((0, 1)) == j
        True

        >>> Vector((0, 1)) == i
        False
        """
        return all(s_coord == o_coord for s_coord, o_coord in zip(self.coordinates, other_point.coordinates))

    def __ne__(self, other_point):
        """Two vectors are not equal if any of their coordinates differ

        >>> Vector((1, 1)) != Vector((1, 1))
        False

        >>> Vector((1, 1)) != Vector((1, 0))
        True

        >>> Vector((1, 0)) != i
        False

        >>> Vector((1, 0)) != j
        True

        >>> Vector((0, 1)) != j
        False

        >>> Vector((0, 1)) != i
        True
        """
        return not (self == other_point)

    def __str__(self):
        return 'Vector({})'.format(', '.join(str(coord) for coord in self.coordinates))


class GeneticSpecimen(object, metaclass=abc.ABCMeta):
    """Class that holds the logic for mating 2 specimens for genetic
    programming
    """

    def __init__(self, dna):
        super().__init__()
        self.dna = dna

    @abc.abstractproperty
    def fitness(self):
        """The fitness level of the genetic specimen, used to determine mating
        hierarchy
        """

    def mate(self, other, use_self_class=True):
        """Randomly combines the DNA of self and the other genetic specimen
        with the possibility of random mutations in the DNA. If use_self_class
        is True, then self.__class__ is used as the child's class, otherwise
        other.__class__ is used.

        ret: A child with either self.__class__ or other.__class__ with the new
             DNA
        """

        cls = self.__class__ if use_self_class else other.__class__

        # Preset the DNA as random
        new_dna = np.random.uniform(size=len(self.dna))

        for i in range(len(new_dna)):
            chance = np.random.randint(3)

            if chance == 0:
                new_dna[i] = self.dna[i]

            elif chance == 1:
                new_dna[i] = other.dna[i]

        return cls(new_dna)


# End Utility Classes
#########################################################################

#########################################################################
# Utility Functions

def format_seconds(seconds):
    """Formats the given number of seconds into '%Hh %Mm %S.%fs' string format

    >>> format_seconds(30)
    '00h 00m 30.000000s'
    >>> format_seconds(60)
    '00h 01m 00.000000s'
    >>> format_seconds(3*60*60 + 51*60 + 12.31)
    '03h 51m 12.310000s'
    """
    return datetime.datetime.utcfromtimestamp(seconds).strftime('%Hh %Mm %S.%fs')


def get_exception_info():
    """ Returns the filename of the exception, and the line number """
    _, exc_obj, exc_tb = sys.exc_info()
    filename = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
    line_number = exc_tb.tb_lineno

    traceback_string = ''.join(traceback.format_exc())

    return filename, line_number, traceback_string

# Utility Functions
#########################################################################


if __name__ == '__main__':
    import docopt
    import doctest
    args = docopt.docopt(__doc__)

    if args['--test']:
        doctest.testmod(
            verbose=args['--verbose'],
            extraglobs={
                'z': Vector((0, 0)),
                'i': Vector((1, 0)),
                'j': Vector((0, 1)),
                'v21': Vector((2, 1)),
                'v_1_2': Vector((-1, -2))
            }
        )
