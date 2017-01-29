# -*- coding: utf-8 -*-

import gym
from gym import wrappers

import numpy as np

import agent

# helper function and dictionaries


def running_average(a, size=100):
    """calculates the running average over array a"""
    ra = []
    ra.append(np.sum(a[:size]))
    for i in range(size, len(a)):
        ra.append(ra[-1] + a[i] - a[i - size])
    return 1. / size * np.array(ra)


translate_move_2d_to_1d = {
    tuple([-1., 0.]): 0,
    tuple([0., 1.]): 1,
    tuple([1., 0.]): 2,
    tuple([0., -1.]): 3,
}

translate_pos_1d_to_2d = {
}

i = 0
for y in np.arange(0., 4., 1.):
    for x in np.arange(0., 4., 1.):
        translate_pos_1d_to_2d[i] = np.array([x, y])
        i += 1

# define parameters

env_name = 'FrozenLake-v0'
initial_pos = [0., 0.]
env_shape = (4, 4)
possible_moves = [np.array(move) for move in translate_move_2d_to_1d.keys()]
n_episodes = 500

# set up environment and recording

env = gym.make(env_name)
# env = wrappers.Monitor(env, '/tmp/frozen-lake-experiment-2')
agent = agent.Agent(initial_pos, env_shape, possible_moves)

# run episodes

episode = 0
episode_reward = []

for episode in range(n_episodes):

    print('episode start.')
    env.reset()

    done = False
    reward = 0.

    agent.reset_position()
    agent.reset_reward()

    obs = 0
    while not done:
        env.render()

        action = agent.step(reward)
        obs, reward, done, _ = env.step(translate_move_2d_to_1d[tuple(action)])

        # since the ice is slippery, we need to correct current
        # position and last move from observation of environment
        agent._pos = translate_pos_1d_to_2d[obs]

    agent._pos = translate_pos_1d_to_2d[obs]

    agent.step(reward - 0.1)

    episode += 1
    episode_reward.append(reward)

    print('episode end. episode {episode}, reward {reward}'.format(episode=episode, reward=np.sum(episode_reward)))

ra = running_average(episode_reward)

print('solved after {n} episodes. maximal reward over 100 episodes: {max_reward}'.format(n=np.where(ra >= 0.78)[0][0], max_reward=np.max(ra)))
