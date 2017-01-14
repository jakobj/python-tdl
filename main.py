# -*- coding: utf-8 -*-

import numpy as np

import universe


n_runs = 10
n_steps = 10000
uni = universe.Universe('grid_world', shape=(4, 4, ))

uni.set_reward([3, 0], 1.)
uni.set_reward([1, 0], -1.)
uni.set_reward([1, 1], -1.)
uni.set_reward([1, 2], -1.)

for i in range(10):
    np.random.seed(1234 + i)

    uni.reset_agent()
    for _ in range(n_steps):
        uni.step()

    print('total reward after {n} steps: {reward}'.format(n=(i + 1) * n_steps, reward=uni._agent._total_reward))

uni.reset_agent()
for _ in range(100):
    uni.show()
    uni.step()
