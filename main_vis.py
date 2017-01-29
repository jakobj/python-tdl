# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import time

import universe


seed = 91231
n_steps_pretraining = 3000
n_steps = 3500

steps_to_reward = 9
max_reward = n_steps / steps_to_reward

# uni = universe.Universe('grid_world', world='world0')
uni = universe.Universe('grid_world', world='2d_world0')
uni.show()

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.set_xlim([0, uni._env._shape[1]])
ax.set_ylim([0, uni._env._shape[0]])
ax_value = fig.add_axes([0.1, 0.1, .8, .8],
                        frameon=False, xticks=[], yticks=[])
ax_value.set_xlim([0, uni._env._shape[1]])
ax_value.set_ylim([0, uni._env._shape[0]])

plt.ion()
plt.show()

uni.plot_env(ax)

np.random.seed(seed)

for _ in range(n_steps_pretraining):
    uni.step()

last_reward = 0.
uni.reset_agent_position()
uni.reset_agent_reward()
for _ in range(n_steps):
    uni.step()
    uni.plot_agent(ax)
    if uni.total_agent_reward() != last_reward:
        uni.plot_value(ax_value)
        last_reward = uni.total_agent_reward()
    plt.pause(0.010)

print('reward:', uni.total_agent_reward())
