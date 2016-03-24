#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Holds the views

Usage:
    view.py -h | --help
    view.py -t | --test [-v|--verbose]

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

import pygame


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class PygameSpriteEntity(pygame.sprite.Sprite, event.EventSource):
    """A displayable entity through the pygame library

    Subscriptions:
        PropertyChangeEvent self.update (source)
    """

    def __init__(self, entity):
        super().__init__()

        self.entity = entity
        self.entity.subscribe_to_source(event.PropertyChangeEvent, self.update)

        self.init()

    def init(self):
        """Extracted constructor code so that the PygameSpriteEntity can be
        re-initialized everytime the its Entity model is updated.
        """

        # Create an image of the block, and fill it with a colour.
        # This could also be an image loaded from the disk.
        self.image = pygame.Surface([self.size // 2, self.size // 2])
        self.image.fill(self.colour)

        # Fetch the rectangle object that has the dimensions of the image
        # Update the position of this object by setting the values of rect.x and rect.y
        self.rect = self.image.get_rect()
        self.rect.center = self.center

    def update(self, prop_change_event):
        """callback function for the PropertyChangeEvent. Reconfigures the sprite
        to make sure that it still looks like the model
        """

        assert self.entity is prop_change_event.obj, 'Entity model\n{}\nchanged to\n{}'.format(
            self.entity,
            prop_change_event.obj
        )
        self.init()

    @property
    def center(self):
        return (
            self.entity.center.coordinates[1],
            self.entity.center.coordinates[0]
        )

    @property
    def size(self):
        return self.entity.size

    @property
    def colour(self):
        return self.entity.colour


class PygameWorld(event.EventSource):
    """A backdrop upon which PygameSpriteEntities are drawn

    Subscriptions:
        ClockTickEvent self.on_tick (global)
        ItemRemovedEvent self.handle_entity_removed (source)
        ItemAddedEvent self.handle_entity_added (source)
    """

    def __init__(self, world, name):
        super().__init__()

        self.event_dispatcher.subscribe_to_all(event.ClockTickEvent, self.render)

        self.world = world
        self.world.subscribe_to_source(event.ItemAddedEvent, self.handle_entity_added)
        self.world.subscribe_to_source(event.ItemRemovedEvent, self.handle_entity_removed)

        self.sprite_entities = {
            entity_model: PygameSpriteEntity(entity_model) for entity_model in self.world.entities
        }

        self.name = name

        pygame.init()
        pygame.display.set_caption(self.name)

        self.window = pygame.display.set_mode(
            (self.world.dimensions[1], self.world.dimensions[0]),
            pygame.DOUBLEBUF
        )
        self.window.set_alpha(None)

        self.background = pygame.Surface(self.window.get_size())
        self.background.fill(WHITE)
        self.window.blit(self.background, (0, 0))

    def handle_entity_added(self, collection_change_event):
        """callback function for ItemAddedEvent. Wraps the item (entity) that was
        as a PygameSpriteEntity and adds it to the pygame group.
        """

        added_entity = collection_change_event.item
        self.sprite_entities[added_entity] = PygameSpriteEntity(added_entity)

    def handle_entity_removed(self, collection_change_event):
        """callback function for ItemRemovedEvent. Removes the item (entity) wrapped
        as a sprite from the pygame group.
        """

        removed_entity = collection_change_event.item
        self.sprite_entities.pop(removed_entity)

    def render(self, clock_tick_event):
        """callback function for ClockTickEvent. Prints a new frame to the pygame window"""
        self.background.fill(WHITE)

        entities = pygame.sprite.Group()
        for entity in self.sprite_entities.values():
            entities.add(entity)

        entities.draw(self.background)

        self.window.blit(self.background, (0, 0))
        pygame.display.update()


if __name__ == '__main__':
    import docopt
    import doctest

    args = docopt.docopt(__doc__)

    if args['--test']:
        doctest.testmod(verbose=args['--verbose'])
