# game_visualization_torch_ffmpeg.py

import json
import os

from dqn_agent_torch import DeepQLearningAgentTorch
from game_environment import SnakeNumpy
from utils import visualize_game

version = 'v17.1'

with open(f'model_config/{version}.json') as f:
    m = json.load(f)
board_size = m['board_size']
frames = m['frames']
max_time_limit = m['max_time_limit']
n_actions = m['n_actions']
obstacles = bool(m['obstacles'])

env = SnakeNumpy(
    board_size=board_size,
    frames=frames,
    max_time_limit=max_time_limit,
    games=1,
    frame_mode=False,
    obstacles=obstacles,
    version=version,
)

agent = DeepQLearningAgentTorch(
    board_size=board_size,
    frames=frames,
    n_actions=n_actions,
    buffer_size=60000,
    gamma=0.99,
    use_target_net=True,
    version=version,
    lr=5e-4,
)

iteration = 200000
agent.load_model(file_path=f'models/{version}', iteration=iteration)

os.makedirs('images', exist_ok=True)
path = f'images/game_visual_{version}_{iteration}.mp4'

visualize_game(env, agent, path=path, debug=False, animate=True, fps=12)
print("Saved:", path)
