#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Holds the models

Usage:
    model.py -h | --help
    model.py -t | --test [-v|--verbose]

Options:
    -h --help       Show this help message
    -t --test       Run the module's unit tests
    -v --verbose    Show all of the output from the unit tests [optional]
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import event
import utils

import abc
import multiprocessing.dummy as mp
import numpy as np


class Entity(event.EventSource, metaclass=abc.ABCMeta):
    """Represents an object with size in n dimensional space, holds the logic for
    adding, subtracting points and calculating the distance between points.

    No Subscriptions

    Emits:
        PropertyChangeEvent
    """

    def __init__(self, food_chain_rank=0, center=None, size=0, colour=0):
        """
        @param food_chain_rank: What is the entity's rank in the food chain,
        entities can only eat things that are lower in rank

        @param center: A Vector object to represent the middle of the entity
        @param size: The size of the entity
        @param colour: The colour of the entity when drawn
        """
        super().__init__()

        self.food_chain_rank = food_chain_rank
        self.center = center
        self.size = size
        self.colour = colour

        # This is used to ensure object stability after passing for multiprocessing
        self._id = id(self)

        # Used to calculate the fitness of the entity
        self.wins = 0
        self.loses = 0

    @classmethod
    def from_coordinates(cls, coordinates, *args, **kwargs):
        center = utils.Vector(coordinates)
        return cls(*args, center=center, **kwargs)

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, new_center):
        self._center = new_center
        self.notify_subscribers(event.PropertyChangeEvent(self))

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, new_size):
        self._size = new_size
        self.notify_subscribers(event.PropertyChangeEvent(self))

    @property
    def colour(self):
        return self._colour

    @colour.setter
    def colour(self, new_colour):
        self._colour = new_colour
        self.notify_subscribers(event.PropertyChangeEvent(self))

    def distance_between(self, other_entity):
        """Measures the distance between 2 entities as the distance between
        their centers
        """
        return self.center.distance_between(other_entity.center)

    def angle_to(self, other_entity):
        """Measures the angle from the 'origin-axis' (0th axis) to the
        other_entity with respect to self
        """

        origin_axis = utils.Vector([0] * len(self.center))
        origin_axis[0] = 1

        return origin_axis.angle_between(other_entity.center - self.center)

    def __str__(self):
        return '{}(center={}, size={})'.format(
            self.__class__.__name__,
            self.center,
            self.size
        )


class Plant(Entity):
    """An actionless entity"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, food_chain_rank=0, **kwargs)


class FoodPellet(Plant):
    """A food pellet that cannot perform any actions"""

    def __init__(self, center):
        super().__init__(center=center, size=10, colour=0)

    def __repr__(self):
        return 'Food'


class EntityComponent(metaclass=abc.ABCMeta):
    """Pieces of an entity, such as an entity's brain, arms, or legs"""

    @classmethod
    @abc.abstractmethod
    def from_dna(cls, dna):
        """Creates an EntityComponent based on the allele found at dna[cls]"""


class Brain(EntityComponent, metaclass=abc.ABCMeta):
    """A brain abstract class that forces the implementation of the
    perform_action method
    """

    def from_dna(cls, dna):
        """Returns a cls without any use of dna since basic brains are
        completely randomized and don't need any instructions from DNA
        """
        return cls()

    @abc.abstractmethod
    def make_decision(self, *args, **kwargs):
        """Returns the components for a unit vector as a set of values between
        -1 and 1, as well as what percentage of the entity's full speed should
        be used; given a set of inputs that describe a situation
        """

    def __str__(self):
        return self.__class__.__name__


class RandomBrain(Brain):
    """Returns a random vector"""

    def make_decision(self, *args, **kwargs):
        return np.random.uniform(-1, 1, 3)


class DumbBrain(Brain):
    """Moves towards the nearest entity that is lower in the food chain"""

    # TODO: remove entity
    def make_decision(self, entity, world_map):
        # Find the closest entity with a lower food_chain_rank
        closest_entity = None
        min_dist = float('inf')

        for row in world_map:
            for cell in row:
                # No need to check if cell is self since it's rank is not less than itself
                if cell is not None and cell.food_chain_rank < entity.food_chain_rank:
                    dist = entity.distance_between(cell)

                    if dist < min_dist:
                        min_dist = dist
                        closest_entity = cell

        dir_vect = closest_entity.center - entity.center
        dir_vect = dir_vect.coordinates / dir_vect.magnitude()

        percent_full_speed = min(1, min_dist / entity.speed)

        return dir_vect[0], dir_vect[1], percent_full_speed


class Animal(Entity):
    """An actionable entity, can move in order to pursue food or to avoid
    predators.
    """

    def __init__(self, food_chain_rank=0, center=None, size=0, colour=0, speed=0, brain=None):
        """
        @param food_chain_rank: What is the entity's rank in the food chain,
        entities can only eat things that are lower in rank

        @param center: Where the animal is located in the world
        @param size: How big is the animal
        @param colour: What colour should be used to draw the animal
        @param speed: The maximum number of spaces that the animal can move at
        once

        @param brain: An object that has a perform_action method that can
        be used to decide what the animal will do
        """

        super().__init__(
            food_chain_rank=food_chain_rank,
            center=center,
            size=size,
            colour=colour
        )

        self.speed = speed
        self.brain = brain if brain is not None else DumbBrain()

    def make_decision(self, *args, **kwargs):
        """Passes the given information to the animal's brain which decides
        what course of action to take
        """

        return self.brain.make_decision(self, *args, **kwargs)

    def perform_action(self, x_dir, y_dir, percent_full_speed):
        mag = self.speed * percent_full_speed

        self.center += utils.Vector((x_dir*mag, y_dir*mag))


class Herbivor(Animal):
    """An animal that only eats food pellets"""

    def __init__(self, center=None, size=20, colour=(0, 255, 0), speed=5, brain=None):
        super().__init__(
            food_chain_rank=1,
            center=center,
            size=size,
            colour=colour,
            speed=speed,
            brain=brain
        )

    def __repr__(self):
        return 'Herb'


class World(event.EventSource):
    """Represents an n dimensional world in which entities can live

    Subscriptions:
        ClockTickEvent self.on_tick (global)
        PropertyChangeEvent self.update (source)

    Emits:
        ItemAddedEvent
        ItemRemovedEvent
        PropertyChangeEvent
    """

    def __init__(self, dimensions):
        """
        @param dimensions: A list of n dimensions that describe this world
        """
        super().__init__()

        self.dimensions = dimensions
        self.entities = set()

        self.event_dispatcher.subscribe_to_all(event.ClockTickEvent, self.on_tick)

    def as_array(self):
        """Creates a 2D matrix consiting of the objects in the world"""

        row = [None] * self.dimensions[1]
        world_map = [row] * self.dimensions[0]
        world_map = np.array(world_map)

        for entity in self.entities:
            center = entity.center.coordinates
            cur_val = world_map[center[0]][center[1]]

            if cur_val is None or cur_val.food_chain_rank < entity.food_chain_rank:
                world_map[center[0]][center[1]] = entity

        return world_map

    def get_random_cell(self):
        return [np.random.randint(dim) for dim in self.dimensions]

    def is_entity_in_bounds(self, entity):
        """Checks to see if the given entity is within this world's dimensions"""

        return all(
            0 <= entity_center < world_dimension for entity_center, world_dimension in zip(
                entity.center,
                self.dimensions
            )
        )

    def add_entity(self, entity):
        """Add the given entity into the world, if the entity's center is
        outside of the world's boundaries then an OutOfBoundsError is thrown.
        If the entity is added successfully, the world is then added as one
        of the subscribers of the entity.

        Emits:
            ItemAddedEvent
            PropertyChangeEvent
        """

        if self.is_entity_in_bounds(entity):
            self.entities.add(entity)
            entity.subscribe_to_source(event.PropertyChangeEvent, self.update)

            self.notify_subscribers(event.ItemAddedEvent(entity, self))
            self.notify_subscribers(event.PropertyChangeEvent(self))

        else:
            raise utils.OutOfBoundsError(entity, self)

    def remove_entity(self, entity):
        """Banish the given entity from this world and remove this world from
        its list of subscribers.

        Emits:
            ItemRemovedEvent
            PropertyChangeEvent
        """
        self.entities.remove(entity)
        self.notify_subscribers(event.ItemRemovedEvent(entity, self))
        self.notify_subscribers(event.PropertyChangeEvent(self))

    def update(self, prop_changed_event):
        """callback function for PropertyChangeEvent. Since the entity has
        changed position we need to make sure that it is still within the
        boundaries of the world.

        Emits:
            PropertyChangeEvent
        """

        changed_entity = prop_changed_event.obj
        if not self.is_entity_in_bounds(changed_entity):
            raise utils.OutOfBoundsError(changed_entity, self)

        self.notify_subscribers(event.PropertyChangeEvent(self))

    def handle_entity_collisions(self):
        """Checks to see which entities have collided and determines who wins"""

        groups = {}
        for entity in self.entities:
            # If this center is already in groups then we have a collision
            if tuple(entity.center) in groups:
                # Get the entity that is already in groups
                first_entity = groups[tuple(entity.center)]

                winning_entity = losing_entity = None

                # Determine which entity wins
                if entity.food_chain_rank < first_entity.food_chain_rank:
                    winning_entity = first_entity
                    losing_entity = entity

                elif entity.food_chain_rank > first_entity.food_chain_rank:
                    winning_entity = entity
                    losing_entity = first_entity

                # If there was a winner
                if winning_entity is not None:
                    # The winner keeps the cell
                    groups[tuple(winning_entity.center)] = winning_entity
                    winning_entity.wins += 1

                    losing_entity.loses += 1

                    # Move the loser to some other square (might not be empty!)
                    new_pos = None
                    while new_pos is None or new_pos == losing_entity.center:
                        new_pos = utils.Vector(self.get_random_cell())

                    losing_entity.center = new_pos

            else:
                # If this cell was empty add it to groups
                groups[tuple(entity.center)] = entity

    def get_entity_decisions(self, entity, entity_id, world_map, q):
        """Separate method for getting the decisions from the individual
        entities to allow for parallelization
        """

        decision = entity.make_decision(world_map)
        q.put((entity_id, decision))

    def handle_entity_decisions(self, entity, decision):
        """Handles the entity's decision by resetting it if there is an error"""

        entity_memento = utils.Memento(entity)
        try:
            entity.perform_action(*decision)

        except utils.OutOfBoundsError:
            utils.warn('{}.handle_entity_decisions'.format(self), '{} tried to go out of bounds...little bugger'.format(entity))
            entity_memento.restore()

    def on_tick(self, clock_tick_event):
        """callback function for the ClockTickEvent. Calls the perform_action
        method in all of the entities in self.world
        """

        world_map = self.as_array()
        m = mp.Manager()
        q = m.Queue()

        args = [(entity, id(entity), world_map, q) for entity in self.entities if hasattr(entity, 'make_decision')]

        with mp.Pool(min(utils.NUM_WORKERS, len(args))) as pool:
            pool.starmap(self.get_entity_decisions, args)

            pool.close()
            pool.join()

        entity_id_map = {}
        while not q.empty():
            entity_id, entity_decision = q.get()
            entity_id_map[entity_id] = entity_decision

        for entity in self.entities:
            entity_decision = entity_id_map.get(id(entity), None)
            if entity_decision is not None:
                self.handle_entity_decisions(entity, entity_decision)

        self.handle_entity_collisions()

    def __str__(self):
        return self.__class__.__name__


if __name__ == '__main__':
    import docopt
    import doctest

    args = docopt.docopt(__doc__)

    if args['--test']:
        doctest.testmod(verbose=args['--verbose'])
