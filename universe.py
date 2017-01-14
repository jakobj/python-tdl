# -*- coding: utf-8 -*-

import itertools
import numpy as np

import agent
import environment


class Universe(object):
    """An agent in an environment."""

    def __init__(self, environment_type, *args, **kwargs):
        if environment_type == 'grid_world':
            self._env = environment.GridWorld(*args, **kwargs)
            self._agent_initial_pos = np.zeros(len(kwargs['shape']))
            self._agent = agent.Agent(self._agent_initial_pos, kwargs['shape'], self._possible_moves_gridworld(kwargs['shape']))
        else:
            raise NotImplementedError('Unknown environment type.')

    def reset_agent(self):
        """resets agent to its initial position"""
        self._agent.set_pos(self._agent_initial_pos)

    def reset_environment(self):
        """sets reward for all positions to zero"""
        self._env.reset()

    def show(self):
        """prints the current status of the universe including agent"""
        uni = self._env.get_text_view()
        uni[tuple(self._agent.get_pos())] = 'x'
        print(uni)
        print()

    def step(self):
        """evolves universe for a single time step"""
        reward = self._env.get_reward(self._agent.get_pos())
        self._agent.step(reward)
        if abs(reward) > 1e-12:  # reset agent upon receiving a reward
            self.reset_agent()

    def set_reward(self, pos, reward):
        """set reward for the specified position. reward can be positive
        ("reward") or negative ("punishment")"""
        self._env.set_reward(pos, reward)

    def _possible_moves_gridworld(self, shape):
        """returns all possible moves in a gridworld, excluding diagonal moves"""
        possible_moves = [m for m in itertools.product([-1., 0., 1.], repeat=len(shape)) if np.dot(m, m) < 2]
        return np.array(possible_moves, dtype=np.float)

    def reset(self):
        """reset universe. resets agent and rewards."""
        self.reset_agent()
        self.reset_environment()
