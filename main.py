# -*- coding: utf-8 -*-

import numpy as np

import universe


seed = 91231
n_trials = 20
n_steps = 1000

steps_to_reward = 9
max_reward = n_steps // steps_to_reward * 5.

uni = universe.Universe('grid_world', world='2d_world0')
uni.show()

reward = []
for i in range(n_trials):
    np.random.seed(1234 + i)

    uni.reset_agent_position()
    uni.reset_agent_reward()
    for _ in range(n_steps):
        uni.step()

    reward.append(uni._agent._total_reward)
    print('total reward after {steps} steps: {reward}/{max_reward}'.format(steps=n_steps, reward=uni._agent._total_reward, max_reward=max_reward))
