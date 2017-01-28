# -*- coding: utf-8 -*-

import collections
import numpy as np


class Agent(object):
    """An agent that produce moves in an arbitrary environment and learns
    from reward."""

    def __init__(self, initial_pos, limit_pos, valid_moves):
        self._pos = np.array(initial_pos, dtype=np.float)
        self._dpos = None
        self._last_pos = collections.deque([])
        self._last_dpos = collections.deque([])
        self._limit_pos = limit_pos
        self._valid_moves = valid_moves
        self._values = collections.defaultdict(lambda: collections.defaultdict(float))
        self._gamma = 0.8
        self._alpha = 1.
        self._total_reward = 0.
        self._lambda_soft_max = 5.

    def _clear_history(self):
        """clears lists of all past states and actions"""
        self._last_pos.clear()
        self._last_dpos.clear()

    def set_pos(self, pos):
        """sets the agent to a certain position"""
        self._clear_history()
        self._pos = pos
        self._dpos = None

    def get_pos(self):
        """returns the position of the agent"""
        return np.array(self._pos, dtype=np.int)

    def reset_reward(self):
        """resets the collected reward to zero"""
        self._total_reward = 0.

    def move(self, dpos):
        """updates the position of the agent to positions + dpos"""

        new_pos = self._pos + dpos
        if np.all(new_pos >= 0.) and np.all(new_pos < self._limit_pos):
            self._last_dpos.appendleft(dpos.copy())
            self._last_pos.appendleft(self._pos.copy())

            self._pos = new_pos
        else:
            raise ValueError('Invalid move.')

    def step(self, reward):
        """evaluate reward from previous move and perform a new move"""
        self._total_reward += reward
        self._dpos = self._select_action()
        self._update_values(reward)
        self.move(self._dpos)

    def _update_values(self, reward):
        """updates predictions based on reward and previous state via Q-learning"""
        if self._last_pos and self._last_dpos and self._dpos is not None:
            lpos = self._last_pos[0]
            ldpos = self._last_dpos[0]

            V_t = self._values[tuple(self._last_pos[0])][tuple(self._last_dpos[0])]
            V_tp = np.max(list(self._values[tuple(self._pos)].values()))
            dV = self._alpha * (reward + self._gamma * V_tp - V_t)

            self._values[tuple(lpos)][tuple(ldpos)] += dV

    def _select_action(self):
        """selects the next move based on current predictions"""
        valid_moves = self._currently_valid_moves()

        rewards = np.zeros(len(valid_moves))

        for i, move in enumerate(valid_moves):
            rewards[i] = self._values[tuple(self._pos)][tuple(move)]

        # convert reward to probabilities via softmax
        p_rewards = self._softmax(rewards)

        return valid_moves[np.random.choice(np.arange(len(valid_moves)), p=p_rewards)]

    def _currently_valid_moves(self):
        """returns valid moves based on current position"""
        valid_moves = []
        for move in self._valid_moves:
            if np.all(self._pos + move >= 0.) and np.all(self._pos + move < self._limit_pos):
                valid_moves.append(move)
        return valid_moves

    def _softmax(self, x):
        max_x = np.max(x)  # substract maximum to avoid exp overflow
        return np.exp(self._lambda_soft_max * (x - max_x)) / np.sum(np.exp(self._lambda_soft_max * (x - max_x)))
