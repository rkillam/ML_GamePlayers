#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Holds the controllers including the main controller for the entire game

Usage:
    controller.py -h | --help
    controller.py -t | --test [-v|--verbose]

Options:
    -h --help       Show this help message
    -t --test       Run the module's unit tests
    -v --verbose    Show all of the output from the unit tests [optional]
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import event
import model
import utils
import view

import collections
import numpy as np
import pygame


class ClockTickController(event.EventSource):
    """Broadcasts a clock tick event as fast as possible.

    Subscriptions:
        QuitEvent self.stop (global)

    Emits:
        ClockTickEvent
    """

    def __init__(self):
        super().__init__()

        self.continue_running = False
        self.event_dispatcher.subscribe_to_all(event.QuitEvent, self.stop)

    def run(self):
        """Starts ticking the clock

        Emits:
            ClockTickEvent
        """

        self.continue_running = True
        while self.continue_running:
            self.notify_subscribers(event.ClockTickEvent())

    def stop(self, quit_event):
        """callback function for the QuitEvent, stops ticking the clock"""
        self.continue_running = False


class PygameEventController(event.EventSource):
    """Receives all relevant pygame generated events and issues custom events
    in their place.

    Subscriptions:
        ClockTickEvent self.check_pygame_events (global)

    Emits:
        QuitEvent
    """

    def __init__(self):
        super().__init__()

        self.event_dispatcher.subscribe_to_all(event.ClockTickEvent, self.check_pygame_events)
        self.pygame_event_map = {
            pygame.QUIT: event.QuitEvent
        }

    def check_pygame_events(self, tick_event):
        """callback function for the ClockTickEvent, checks all pygame events
        to see if one that has been wrapped was issued. If so, then the wrapped
        event is issued.

        Emits:
            QuitEvent
        """

        for event in pygame.event.get():
            if event.type in self.pygame_event_map:
                self.notify_subscribers(self.pygame_event_map[event.type]())


if __name__ == '__main__':
    import docopt
    import doctest

    args = docopt.docopt(__doc__)

    if args['--test']:
        doctest.testmod(verbose=args['--verbose'])
