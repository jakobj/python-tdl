# -*- coding: utf-8 -*-

import collections
import numpy as np


class Agent(object):
    """An agent that produce moves in an arbitrary environment and learns
    from reward."""

    def __init__(self, initial_pos, limit_pos, possible_moves):
        self._pos = np.array(initial_pos, dtype=np.float)
        self._last_pos = collections.deque([])
        self._last_dpos = collections.deque([])
        self._limit_pos = limit_pos
        self._possible_moves = possible_moves
        self._exclude_out_of_bounds_moves = True
        self._values = collections.defaultdict(lambda: collections.defaultdict(float))
        self._gamma = 0.8
        self._alpha = .25
        self._total_reward = 0.
        self._lambda_soft_max = 0.
        self._stubbornness = 0.05

        self._agent_ln = None  # reference to line object for plotting position

        # extensions
        self._adaptive_lambda_softmax = True

    def _clear_history(self):
        """clears lists of all past states and actions"""
        self._last_pos.clear()
        self._last_dpos.clear()

    def set_pos(self, pos):
        """sets the agent to a certain position"""
        self._clear_history()
        self._pos = pos

    def get_pos(self):
        """returns the position of the agent"""
        return np.array(self._pos, dtype=np.int)

    def reset_reward(self):
        """resets the collected reward to zero"""
        self._total_reward = 0.

    def move(self, dpos):
        """updates the position of the agent to positions + dpos"""

        new_pos = self._pos + dpos
        self._last_dpos.appendleft(dpos.copy())
        self._last_pos.appendleft(self._pos.copy())
        self._pos = new_pos

    def step(self, reward):
        """evaluate reward from previous move and perform a new move"""
        self._total_reward += reward
        dpos = self._select_action()
        self._update_values(reward)
        self.move(dpos)
        return dpos

    def _update_values(self, reward):
        """updates predictions based on reward and previous state via Q-learning"""
        if self._last_pos and self._last_dpos:
            lpos = self._last_pos[0]
            ldpos = self._last_dpos[0]

            V_t = self._values[tuple(lpos)][tuple(ldpos)]
            V_tp = np.max(list(self._values[tuple(self._pos)].values()))
            dV = self._alpha * (reward + self._gamma * V_tp - V_t)

            self._values[tuple(lpos)][tuple(ldpos)] += dV

            if self._adaptive_lambda_softmax:
                # update exploration/exploitation behviour according to
                # change of value, i.e., surprise
                self._update_lambda_softmax(dV)

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
        if self._exclude_out_of_bounds_moves:
            valid_moves = []
            for move in self._possible_moves:
                if np.all(self._pos + move >= 0.) and np.all(self._pos + move < self._limit_pos):
                    valid_moves.append(move)
            return valid_moves
        else:
            return self._possible_moves

    def _softmax(self, x):
        max_x = np.max(x)  # substract maximum to avoid exp overflow
        return np.exp(self._lambda_soft_max * (x - max_x)) / np.sum(np.exp(self._lambda_soft_max * (x - max_x)))


    def plot(self, ax):
        pos = self._pos[::-1] + 0.5
        if self._agent_ln is None:
            self._agent_ln, = ax.plot(pos, ls='', marker='o', color='r', markersize=20)
        else:
            self._agent_ln.set_data(pos)

    def plot_value(self, ax):
        ax.cla()
        for pos in self._values:
            for dpos in self._possible_moves:
                val = np.round(self._values[pos][tuple(dpos)], 2)
                ax.text(pos[1] + 0.5 + 0.15 * dpos[1], pos[0] + 0.5 + 0.15 * dpos[0], val, ha='center', va='center', fontsize=15)

    def _update_lambda_softmax(self, dV):
        """updates the scaling factor in softmax balancing the amount of sub-optimal moves"""
        self._lambda_soft_max += (0.5 - 1. / (1. + np.exp(-(abs(dV) - self._stubbornness))))
        self._lambda_soft_max = np.max([0., self._lambda_soft_max])
