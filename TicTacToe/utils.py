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

import abc
import collections
import copy
import datetime
import logging
import logging.config
import logging.handlers
import numpy as np
import weakref


#########################################################################
# Logging

# Set configuration file for the logger
logging.config.fileConfig("logging.conf")

# Get our logger as specified in logging.conf
_logger = logging.getLogger("ttt")
_log_format = '{obj} - {msg}'


DEBUG = logging.DEBUG
WARNING = logging.WARNING
ERROR = logging.ERROR


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
