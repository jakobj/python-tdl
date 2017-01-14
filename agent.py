# -*- coding: utf-8 -*-

import collections
import numpy as np


class Agent(object):
    """An agent that produce moves in an arbitrary environment and learns
    from reward."""

    def __init__(self, initial_pos, limit_pos, valid_moves):
        self._pos = np.array(initial_pos, dtype=np.float)
        self._dpos = None
        self._last_pos = None
        self._last_dpos = None
        self._limit_pos = limit_pos
        self._valid_moves = valid_moves
        self._values = collections.defaultdict(lambda: collections.defaultdict(float))
        self._gamma = 0.1
        self._alpha = 0.25
        self._total_reward = 0.
        self._lambda_soft_max = 10.

    def set_pos(self, pos):
        self._pos = pos

    def get_pos(self):
        return np.array(self._pos, dtype=np.int)

    def move(self, dpos):
        """updates the position of the agent to positions + dpos"""

        new_pos = self._pos + dpos
        if np.all(new_pos >= 0.) and np.all(new_pos < self._limit_pos):
            self._last_dpos = dpos
            self._last_pos = self._pos.copy()
            self._pos = new_pos
        else:
            raise ValueError('Invalid move.')

    def step(self, reward):
        """evaluate reward from previous move and perform a new move"""
        self._total_reward += reward
        self._update_values(reward)
        self._dpos = self._select_action()
        self.move(self._dpos)

    def _update_values(self, reward):
        """updates predictions based on reward and previous state"""
        if self._last_pos is not None and self._dpos is not None:
            V_t = self._values[tuple(self._last_pos)][tuple(self._last_dpos)]
            V_tp = self._gamma * self._values[tuple(self._pos)][tuple(self._dpos)]
            dV = self._alpha * (reward + V_tp - V_t)
            self._values[tuple(self._last_pos)][tuple(self._last_dpos)] += dV

    def _select_action(self):
        """selects the next move based on current predictions"""
        valid_moves = self._currently_valid_moves()

        if self._last_pos is None:
            valid_moves[np.random.choice(np.arange(len(valid_moves)))]

        rewards = np.zeros(len(valid_moves))

        for i, move in enumerate(valid_moves):
            rewards[i] = self._values[tuple(self._pos)][tuple(move)]

        # convert reward to probabilities via softmax
        p_rewards = np.exp(self._lambda_soft_max * rewards) / np.sum(np.exp(self._lambda_soft_max * rewards))
        return valid_moves[np.random.choice(np.arange(len(valid_moves)), p=p_rewards)]

    def _currently_valid_moves(self):
        """returns valid moves based on current position"""
        valid_moves = []
        for move in self._valid_moves:
            if np.all(self._pos + move >= 0.) and np.all(self._pos + move < self._limit_pos):
                valid_moves.append(move)
        return valid_moves
