# -*- coding: utf-8 -*-

import numpy as np


class GridWorld(object):
    """An n-dimensional discrete set of positions that can hold reward."""

    def __init__(self, shape):
        self._shape = shape
        self._world = np.zeros(self._shape)

    def set_reward(self, pos, reward):
        self._world[tuple(pos)] = reward

    def get_reward(self, pos):
        return self._world[tuple(pos)]

    def get_text_view(self):
        """returns a printable view of the environment"""
        return np.array(self._world, dtype=str)

    def get_state(self):
        return self._world

    def reset(self):
        """resets rewards, i.e., all positions to zero reward"""
        self._world = np.zeros(self._shape)
