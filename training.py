'''
script for training the agent for snake using various methods
(Updated to use PyTorch DeepQLearningAgentTorch)
'''

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from tqdm import tqdm
from collections import deque
import pandas as pd
import time
from utils import play_game, play_game2
from game_environment import Snake, SnakeNumpy
import json
import torch
import random

from dqn_agent_torch import DeepQLearningAgentTorch

# ----------------------------------------------------------------------
# Global setup
# ----------------------------------------------------------------------

# seeds
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

version = 'v17.1'

# get training configurations
with open('model_config/{:s}.json'.format(version), 'r') as f:
    m = json.loads(f.read())
    board_size = m['board_size']
    frames = m['frames']  # keep frames >= 2
    max_time_limit = m['max_time_limit']
    supervised = bool(m['supervised'])
    n_actions = m['n_actions']
    obstacles = bool(m['obstacles'])
    buffer_size = m['buffer_size']

# define no of episodes, logging frequency
episodes = 2 * (10**5)
log_frequency = 500
games_eval = 8

# ----------------------------------------------------------------------
# Setup PyTorch Deep Q-Learning agent
# ----------------------------------------------------------------------
agent = DeepQLearningAgentTorch(
    board_size=board_size,
    frames=frames,
    n_actions=n_actions,
    buffer_size=buffer_size,
    gamma=0.99,          # match original config
    use_target_net=True,
    version=version,
    lr=5e-4,             # RMSprop(0.0005) like the TF version
)

agent_type = 'DeepQLearningAgent'
print('Agent is {:s}'.format(agent_type))

# ----------------------------------------------------------------------
# Epsilon / reward / training config (DQN only)
# ----------------------------------------------------------------------
epsilon, epsilon_end = 1, 0.01
reward_type = 'current'
sample_actions = False
n_games_training = 8 * 16
decay = 0.97

if supervised:
    # lower the epsilon since some starting policy has already been trained
    epsilon = 0.01
    # load the existing model from a supervised method or other pretrained model
    agent.load_model(file_path='models/{:s}'.format(version))

# ----------------------------------------------------------------------
# Initial buffer filling for DQN
# ----------------------------------------------------------------------
# play some games initially to fill the buffer
# or load from an existing buffer (supervised)
if supervised:
    try:
        agent.load_buffer(file_path='models/{:s}'.format(version), iteration=1)
    except FileNotFoundError:
        pass
else:
    # setup the environment
    games = 512
    env_init = SnakeNumpy(
        board_size=board_size,
        frames=frames,
        max_time_limit=max_time_limit,
        games=games,
        frame_mode=True,
        obstacles=obstacles,
        version=version,
    )
    ct = time.time()
    _ = play_game2(
        env_init,
        agent,
        n_actions,
        n_games=games,
        record=True,
        epsilon=epsilon,
        verbose=True,
        reset_seed=False,
        frame_mode=True,
        total_frames=games * 64,
    )
    print('Playing {:d} frames took {:.2f}s'.format(games * 64, time.time() - ct))

# ----------------------------------------------------------------------
# Training and evaluation environments
# ----------------------------------------------------------------------
env = SnakeNumpy(
    board_size=board_size,
    frames=frames,
    max_time_limit=max_time_limit,
    games=n_games_training,
    frame_mode=True,
    obstacles=obstacles,
    version=version,
)

env2 = SnakeNumpy(
    board_size=board_size,
    frames=frames,
    max_time_limit=max_time_limit,
    games=games_eval,
    frame_mode=True,
    obstacles=obstacles,
    version=version,
)

# ----------------------------------------------------------------------
# Training loop (DQN only)
# ----------------------------------------------------------------------
model_logs = {
    'iteration': [],
    'reward_mean': [],
    'length_mean': [],
    'games': [],
    'loss': [],
}

for index in tqdm(range(episodes)):
    # --------------------------------------------------------------
    # Deep Q-Learning: fill buffer and train in small steps
    # --------------------------------------------------------------
    _, _, _ = play_game2(
        env,
        agent,
        n_actions,
        epsilon=epsilon,
        n_games=n_games_training,
        record=True,
        sample_actions=sample_actions,
        reward_type=reward_type,
        frame_mode=True,
        total_frames=n_games_training,
        stateful=True,
    )

    loss = agent.train_agent(
        batch_size=64,
        num_games=n_games_training,
        reward_clip=True,
    )

    # --------------------------------------------------------------
    # Logging / evaluation
    # --------------------------------------------------------------
    if (index + 1) % log_frequency == 0:
        # evaluate current policy
        current_rewards, current_lengths, current_games = play_game2(
            env2,
            agent,
            n_actions,
            n_games=games_eval,
            epsilon=-1,           # -1 => greedy evaluation (no exploration)
            record=False,
            sample_actions=False,
            frame_mode=True,
            total_frames=-1,
            total_games=games_eval,
        )

        model_logs['iteration'].append(index + 1)
        model_logs['reward_mean'].append(
            round(int(current_rewards) / current_games, 2)
        )
        model_logs['length_mean'].append(
            round(int(current_lengths) / current_games, 2)
        )
        model_logs['games'].append(current_games)
        model_logs['loss'].append(loss)

        pd.DataFrame(model_logs)[
            ['iteration', 'reward_mean', 'length_mean', 'games', 'loss']
        ].to_csv('model_logs/{:s}.csv'.format(version), index=False)

    # --------------------------------------------------------------
    # Target network update + model saving + epsilon decay
    # --------------------------------------------------------------
    if (index + 1) % log_frequency == 0:
        agent.update_target_net()
        agent.save_model(
            file_path='models/{:s}'.format(version),
            iteration=(index + 1),
        )
        # keep some epsilon alive for training
        epsilon = max(epsilon * decay, epsilon_end)
