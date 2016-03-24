#!/usr/bin/python3

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import numpy as np

import pygame
from pygame.locals import *

import Entities


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def product(iterable):
    p = 1
    for i in iterable:
        p *= i
    return p


class World(object):
    def __init__(self, num_cells, dimensions=None, virus_concentration=0.0001, food_concentration=0.002):
        if dimensions is None:
            self.dimensions = (1350, 670)

        else:
            self.dimensions = dimensions

        self._running = True
        self._display_surf = None

        self.cells = None
        self.foods = None
        self.viruses = None
        self.entities = None

        self.num_cells = num_cells
        self.num_viruses = 0  #int(self.num_spaces * virus_concentration)
        self.num_foods = 5  #int(self.num_spaces * food_concentration)

    @property
    def rect(self):
        return self._display_surf.get_rect()

    @property
    def num_spaces(self):
        return product(self.dimensions)

    @property
    def size(self):
        return self.dimensions[0], self.dimensions[1]

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode(
            self.size,
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self._display_surf.set_alpha(None)

        self._running = True

        self.cells = pygame.sprite.Group()
        self.foods = pygame.sprite.Group()
        self.viruses = pygame.sprite.Group()
        self.entities = [self.cells, self.foods, self.viruses]

        coords = [range(dim) for dim in self.dimensions]
        free_spaces = [Entities.Point(pt) for pt in itertools.product(*coords)]

        for num, entity_cls, group in [
            (self.num_cells, Entities.CellBasicBrain, self.cells),
            (self.num_foods, Entities.Food, self.foods),
            (self.num_viruses, Entities.Virus, self.viruses)]:

            group.add([self.fill_random_space(entity_cls, free_spaces) for _ in range(num)])

    def fill_random_space(self, entity_cls, free_spaces=None):
        if free_spaces is not None:
            new_point = free_spaces.pop(np.random.randint(len(free_spaces)))

        else:
            new_point = Entities.Point([np.random.random() * dim for dim in self.dimensions])

        entity = entity_cls(self, new_point)
        return entity

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

    def on_render(self):
        self._display_surf.fill(WHITE) #This clears the screen on each redraw

        for group in self.entities:
            group.draw(self._display_surf)

        pygame.display.update()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)

            self.on_loop()
            self.on_render()

        self.on_cleanup()

    def on_loop(self):
        for group in self.entities:
            group.update()

        # Detect all food collisions
        pygame.sprite.groupcollide(
            self.cells,
            self.foods,
            False,
            True,
            Entities.Entity.overlap
        )

        # Detect all virus collisions
        pygame.sprite.groupcollide(
            self.cells,
            self.viruses,
            False,
            True,
            Entities.Entity.overlap
        )

        pygame.sprite.groupcollide(
            self.cells,
            self.cells,
            False,
            True,
            Entities.Entity.overlap
        )

        while len(self.foods.sprites()) < self.num_foods:
            self.foods.add(self.fill_random_space(Entities.Food))

        while len(self.viruses.sprites()) < self.num_viruses:
            self.viruses.add(self.fill_random_space(Entities.Virus))


if __name__ == '__main__':
    w = World(10, dimensions=(500, 500))
    w.on_execute()
