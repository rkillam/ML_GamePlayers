#!/usr/bin/python3

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import math
import numpy as np
import pygame
from pygame.locals import *

# import sknn.mlp


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def random_colour():
    return (
        np.random.randint(256),
        np.random.randint(256),
        np.random.randint(256)
    )


class Point(tuple):
    def __init__(self, coordinates):
        self.coordinates = np.array(coordinates)

    @property
    def x(self):
        return self.coordinates[0]

    @property
    def y(self):
        return self.coordinates[1]

    @x.setter
    def x(self, new_x):
        self.coordinates[0] = new_x

    @y.setter
    def y(self, new_y):
        self.coordinates[1] = new_y

    def distance(self, other_point):
        return np.linalg.norm(self.coordinates - other_point.coordinates)

    def __add__(self, other_point):
        try:
            new_pt = self.coordinates + other_point.coordinates

        except AttributeError:
            new_pt = self.coordinates + np.array(other_point)

        return Point(tuple(new_pt))

    def __sub__(self, other_point):
        neg_other_point = [-coord for coord in other_point]

        return self + neg_other_point

    def __getitem__(self, idx):
        return self.coordinates[idx]

    def __setitem__(self, idx, coord):
        self.coordinates[idx] = coord

    def __iter__(self):
        for coord in self.coordinates:
            yield coord

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other_point):
        return self.distance(other_point) <= 0

    def __ne__(self, other_point):
        return not (self == other_point)

    def __str__(self):
        return 'Point({})'.format(', '.join(str(coord) for coord in self.coordinates))


class Entity(pygame.sprite.Sprite):
    def __init__(self, world, pos, size, colour):
        super(Entity, self).__init__()

        self._pos = pos
        self._colour = colour
        self.org_colour = colour
        self.world = world

        # Create an image of the block, and fill it with a color.
        # This could also be an image loaded from the disk.
        self.image = pygame.Surface([size // 2, size // 2])
        self.image.fill(self.colour)

        # Fetch the rectangle object that has the dimensions of the image
        # Update the position of this object by setting the values of rect.x and rect.y
        self.rect = self.image.get_rect()

        self.rect.center = self.pos

    def reset_colour(self):
        self.colour = self.org_colour

    @property
    def colour(self):
        return self._colour

    @colour.setter
    def colour(self, new_colour):
        self._colour = new_colour
        self.image.fill(self._colour)

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, new_pos):
        if new_pos.x < 0:
            new_pos.x = 0
        elif self.world.rect.width < new_pos.x:
            new_pos.x = self.world.rect.width

        if new_pos.y < 0:
            new_pos.y = 0
        elif self.world.rect.height < new_pos.y:
            new_pos.y = self.world.rect.height

        self._pos = new_pos
        self.rect.center = new_pos

    @property
    def size(self):
        return self.rect.width

    @size.setter
    def size(self, new_size):
        self.image = pygame.transform.scale(self.image, (new_size, new_size))
        self.image.fill(self.colour)

        # Fetch the rectangle object that has the dimensions of the image
        # Update the position of this object by setting the values of rect.x and rect.y
        self.rect = self.image.get_rect()

        self.rect.center = self.pos

    def distance(self, other_entity):
        return self.pos.distance(other_entity.pos)

    def angle_from_x(self, other_entity):
        delta_x = other_entity.pos.x - self.pos.x
        delta_y = other_entity.pos.y - self.pos.y

        return math.atan2(delta_y, delta_x)

    def overlap(self, other_entity):
        distance = self.distance(other_entity)
        overlap = distance < (self.size + other_entity.size) / 2

        if overlap:
            return self.collide(other_entity)

        else:
            return False

    def collide(self, other_entity):
        if self.size > other_entity.size:
            return self.win(other_entity)

    def win(self, other_entity):
        pass

    def lose(self, other_entity):
        other_entity.size += self.size


"""
class PygameSprite(pygame.sprite.Sprite, Entity):
    def __init__(self, center, size, colour):
        pygame.sprite.Sprite.__init__(self)
        Entity.__init__(self, center, size)

        self._colour = colour

        # Create an image of the block, and fill it with a colour.
        # This could also be an image loaded from the disk.
        self.image = pygame.Surface([self.size // 2, self.size // 2])
        self.image.fill(self.colour)

        # Fetch the rectangle object that has the dimensions of the image
        # Update the position of this object by setting the values of rect.x and rect.y
        self.rect = self.image.get_rect()
        self.rect.center = self.center

    @property
    def colour(self):
        return self._colour

    @colour.setter
    def colour(self, new_colour):
        self._colour = new_colour
        self.image.fill(self._colour)

    @property
    def center(self):
        return self.center

    @center.setter
    def center(self, new_center):
        super().center = new_center
        self.rect.center = self.center

    @property
    def size(self):
        return self.size

    @size.setter
    def size(self, new_size):
        Entity.size = new_size
        self.image = pygame.transform.scale(self.image, (self.size, self.size))
        self.image.fill(self.colour)

        # Fetch the rectangle object that has the dimensions of the image
        # Update the position of this object by setting the values of rect.x and rect.y
        self.rect = self.image.get_rect()
        self.rect.center = self.center
"""


class Food(Entity):
    def __init__(self, world, pos):
        super(Food, self).__init__(world, pos, 5, GREEN)

    def lose(self, other_entity):
        other_entity.size += 1


class Virus(Entity):
    def __init__(self, world, pos):
        super(Virus, self).__init__(world, pos, 50, RED)

    def lose(self, other_entity):
        new_size = max(10, other_entity.size // 10)
        other_entity.size = new_size


class FOV(Entity):
    def win(self, other_entity):
        return True


class Cell(Entity):
    def __init__(self, world, start_pos, start_size=10):
        super(Cell, self).__init__(world, start_pos, start_size, BLUE)  # random_colour())
        self.updates = 0

    @property
    def speed(self):
        return 100 / self.size

    def highlight_fov(self):
        fov_sprite = FOV(self.world, self.pos, self.size * 1000, WHITE)

        viewable_entities = []
        for entity_group in self.world.entities:
            for sprite in entity_group.sprites():
                sprite.reset_colour()

            viewable_sprites = pygame.sprite.spritecollide(
                fov_sprite,
                entity_group,
                False,
                Entity.overlap
            )

            for sprite in viewable_sprites:
                if sprite is not self:
                    viewable_entities.append(sprite)
                    sprite.colour = (0, 0, 0)

        return viewable_entities

    def move(self, theta=None, magnitude=None):
        if theta is None:
            theta = np.random.random() * 2*np.pi

        if magnitude is None:
            magnitude = np.random.random() * self.speed

        delta_x = np.cos(theta) * magnitude
        delta_y = np.sin(theta) * magnitude

        self.pos += Point([delta_x, delta_y])

    def update(self):
        self.highlight_fov()
        self.move()
        super(Cell, self).update()

    def win(self, other_entity):
        other_entity.lose(self)
        return True


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


class CellNN(Cell):
    def __init__(self, *args, **kwargs):
        super(CellNN, self).__init__(*args, **kwargs)
        self.nn = sknn.mlp.Classifier(
            layers=[
                sknn.mlp.Layer('Sigmoid', units=3),
                sknn.mlp.Layer('Softmax', units=1)
            ]
        )

    def update(self):
        viewable_entities = self.highlight_fov()

        if viewable_entities:
            features = np.array([[1, entity.size, self.distance(entity)] for entity in viewable_entities])
            h = self.nn.predict_proba(features)

            idx = np.argmax(h)
            target_entity = viewable_entities[idx]

            theta = self.angle_from_x(target_entity)
            magnitude = min(self.speed, self.distance(target_entity))

        else:
            theta = None
            magnitude = None

        self.move(theta=theta, magnitude=magnitude)

        super(CellNN, self).update()


class CellBasicBrain(Cell):
    def __init__(self, *args, **kwargs):
        super(CellBasicBrain, self).__init__(*args, **kwargs)

    def update(self):
        viewable_entities = self.highlight_fov()

        if viewable_entities:
            features = np.array([[entity.size, 1/self.distance(entity)] for entity in viewable_entities]).sum(axis=1)
            h = sigmoid(features)

            idx = np.argmax(h)
            target_entity = viewable_entities[idx]

            theta = self.angle_from_x(target_entity)
            magnitude = min(self.speed, self.distance(target_entity))

        else:
            theta = None
            magnitude = None

        self.move(theta=theta, magnitude=magnitude)

        super(CellBasicBrain, self).update()
