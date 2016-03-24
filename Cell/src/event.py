#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Holds the events and event manager which act as the main point of
interaction between the model, view, and controller.

Usage:
    event.py -h | --help
    event.py -t | --test [-v|--verbose]

Options:
    -h --help       Show this help message
    -t --test       Run the module's unit tests
    -v --verbose    Show all of the output from the unit tests [optional]
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import utils

import collections


class Event(object):
    """Parent event class"""

    def __init__(self):
        super().__init__()


class ClockTickEvent(Event):
    """Event to signify a clock tick which represents the passage of time"""


class PropertyChangeEvent(Event):
    """Event to signify when an object has changed one or more of its
    properties
    """

    def __init__(self, obj):
        super().__init__()
        self.obj = obj


class CollectionUpdateEvent(PropertyChangeEvent):
    """Event to signify when an item in a collection has been updated"""

    def __init__(self, item, collection):
        self.item = item
        self.collection = collection


class ItemAddedEvent(CollectionUpdateEvent):
    """Event to signify when an item has been added to a collection"""


class ItemRemovedEvent(CollectionUpdateEvent):
    """Event to signify when an item has been removed from a collection"""


class QuitEvent(Event):
    """Event to signify that the game should be closed"""


class EventDispatcher(object):
    """Acts as a mediator between all objects that either issue or listen for
    different events.
    """

    def __init__(self):
        self.subscribers = collections.defaultdict(list)

    def subscribe_to_all(self, event_cls, call_back):
        """Subscribes the given call_back function to all events of the given
        class, regardless of the event's source
        """

        self.subscribers[event_cls].append(call_back)

    def notify_subscribers(self, event):
        """Notifies all subscribers that the current event has occured"""

        for subscriber_callback in self.subscribers[event.__class__]:
            subscriber_callback(event)


# EventDispatcher object that all classes will use, equivalent to making
# EventDispatcher a singleton class
event_dispatcher = EventDispatcher()


class EventSource(object):
    """Parent class for all classes that want to emit events"""

    def __init__(self):
        self.event_dispatcher = event_dispatcher
        self.source_subscribers = collections.defaultdict(list)

    def subscribe_to_source(self, event_cls, subscriber_callback):
        """Subscribes the given callback function to all events of the given
        class, but only if they are emitted from the current object
        """

        self.source_subscribers[event_cls].append(subscriber_callback)

    def notify_subscribers(self, event):
        """Notifies all source subscribers, and then it notifies all global
        subscribers
        """

        # Notify source specific subscribers first
        for subscriber_callback in self.source_subscribers[event.__class__]:
            subscriber_callback(event)

        # Then notify the global subscribers
        self.event_dispatcher.notify_subscribers(event)
