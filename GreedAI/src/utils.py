#!/usr/bin/python3
# -*- coding: utf-8 -*-


class Registry(object):
    """Allows the registration of various objects under different names"""

    def __init__(self, *, name=None):
        self.name = name
        self._registry = {}

    def register(self, name):
        """Returns a decorator function that will register the decorated
        funtion under the given name
        """

        def decorator(f):
            self._registry[name] = f
            return f

        return decorator

    def __getitem__(self, key):
        return self._registry[key]

    @property
    def keys(self):
        return self._registry.keys()
