# -*- coding: utf-8 -*-

import itertools
import numpy as np


class GridWorld(object):
    """An n-dimensional discrete set of positions that can hold reward."""

    def __init__(self, shape=None, world=None):
        if world is None:
            assert(world is None)
            self._world = np.zeros(self._shape)
        else:
            assert(shape is None)
            self._world = np.loadtxt('./gridworlds/{world}.csv'.format(world=world), delimiter=',')
            if len(np.shape(self._world)) == 1:
                self._world = np.array([self._world])

        self._shape = np.shape(self._world)

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

    def possible_moves(self):
        """returns all possible moves in a gridworld, excluding diagonal moves"""
        possible_moves = [move for move in itertools.product([-1., 0., 1.], repeat=len(self._shape)) if np.dot(move, move) < 2]
        return np.array(possible_moves, dtype=np.float)

    def shape(self):
        return self._shape

    def initial_position(self):
        return np.zeros(len(self._shape))

    def plot(self, ax, vmin=-5., vmax=5.):
        ax.pcolormesh(self._world, cmap='RdBu', vmin=vmin, vmax=vmax)
