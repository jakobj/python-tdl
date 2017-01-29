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
            self.create_new_gridworld_agent()
        else:
            raise NotImplementedError('Unknown environment type.')

    def create_new_gridworld_agent(self):
        """creates a new agent for a gridworld"""
        self._agent = agent.Agent(self._env.initial_position(), self._env.shape(), self._env.possible_moves())

    def reset_agent_position(self):
        """resets agent to its initial position"""
        self._agent.set_pos(self._env.initial_position())

    def reset_agent_reward(self):
        """resets agents reward"""
        self._agent.reset_reward()

    def reset_environment(self):
        """sets reward for all positions to zero"""
        self._env.reset()

    def step(self):
        """evolves universe for a single time step"""
        reward = self._env.get_reward(self._agent.get_pos())
        self._agent.step(reward)
        if reward > 1e-12:  # reset agent upon receiving a reward
            self.reset_agent_position()

    def set_reward(self, pos, reward):
        """set reward for the specified position. reward can be positive
        ("reward") or negative ("punishment")"""
        self._env.set_reward(pos, reward)

    def reset(self):
        """reset universe. resets agent and rewards."""
        self.reset_agent_position()
        self.reset_environment()

    def total_agent_reward(self):
        """returns total reward gathered by agent"""
        return self._agent._total_reward

    def plot_agent(self, ax):
        """plots the current position of the agent to the given axis"""
        self._agent.plot(ax)

    def plot_env(self, ax):
        self._env.plot(ax)

    def plot_value(self, ax):
        self._agent.plot_value(ax)

    def show(self):
        """prints the current status of the universe including agent"""
        uni = self._env.get_text_view()
        uni[tuple(self._agent.get_pos())] = 'x'
        print(uni)
        print()

