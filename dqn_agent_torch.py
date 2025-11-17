import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from replay_buffer import ReplayBufferNumpy


class DQNCNN(nn.Module):
    """
    Simple CNN for Snake DQN, matching v17.1 style:
    conv_filters: "16, 32, 64 filters of (3,3), (3,3), (6,6)"
    dense: 64
    board_size: 10
    frames: 2
    """
    def __init__(self, board_size: int, n_frames: int, n_actions: int):
        super().__init__()
        self.board_size = board_size
        self.n_frames = n_frames
        self.n_actions = n_actions

        # Input: [B, C, H, W] with C = n_frames
        self.conv1 = nn.Conv2d(n_frames, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=6, stride=1)

        # conv1: 10x10 -> 8x8
        # conv2: 8x8 -> 6x6
        # conv3: 6x6 -> 1x1
        self.fc1 = nn.Linear(64 * 1 * 1, 64)
        self.fc_out = nn.Linear(64, n_actions)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)   # flatten
        x = self.relu(self.fc1(x))
        x = self.fc_out(x)          # [B, n_actions], Q-values
        return x


class DeepQLearningAgentTorch:
    """
    PyTorch implementation of DeepQLearningAgent.

    Designed to be API-compatible with what training.py/utils.py expect:
      - move()
      - get_action_proba()
      - add_to_buffer()
      - train_agent()
      - update_target_net()
      - save_model() / load_model()
      - save_buffer() / load_buffer()
      - get_gamma() / get_buffer_size()
    """

    def __init__(
        self,
        board_size: int = 10,
        frames: int = 2,
        buffer_size: int = 60000,
        gamma: float = 0.99,
        n_actions: int = 4,
        use_target_net: bool = True,
        version: str = "v17.1",
        lr: float = 5e-4,
    ):
        self._board_size = board_size
        self._n_frames = frames
        self._input_shape = (board_size, board_size, frames)
        self._n_actions = n_actions
        self._buffer_size = buffer_size
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._version = version

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Replay buffer (uses numpy, no TF)
        self._buffer = ReplayBufferNumpy(
            buffer_size=self._buffer_size,
            board_size=self._board_size,
            frames=self._n_frames,
            actions=self._n_actions,   # ✅ correct keyword
        )

        # Networks
        self.policy_net = DQNCNN(board_size, frames, n_actions).to(self.device)
        self.target_net = DQNCNN(board_size, frames, n_actions).to(self.device)
        self.update_target_net()  # copy initial weights

        # Optimizer + loss (Huber is SmoothL1)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    # ------------------------------------------------------------------
    # Small utilities / compatibility helpers
    # ------------------------------------------------------------------

    def get_gamma(self):
        return self._gamma

    def get_buffer_size(self):
        return self._buffer.get_current_size()

    def _normalize_board(self, board: np.ndarray) -> np.ndarray:
        """
        Normalize board values before feeding into the network.
        Original config uses 'divide by 4.0' as board_normalize.
        """
        board = board.astype(np.float32)
        return (board / 4.0).astype(np.float32)

    def _prepare_input(self, board: np.ndarray) -> torch.Tensor:
        """
        board: numpy array with shape
            [H, W, C] or [B, H, W, C]
        Return: torch.Tensor [B, C, H, W] on self.device
        """
        if board.ndim == 3:
            board = board.reshape((1,) + self._input_shape)
        board = self._normalize_board(board.copy())
        # to tensor
        x = torch.from_numpy(board).float().to(self.device)  # [B, H, W, C]
        x = x.permute(0, 3, 1, 2).contiguous()               # [B, C, H, W]
        return x

    def _get_model_outputs(self, board: np.ndarray, model=None) -> np.ndarray:
        """
        Helper: run a forward pass and return Q-values as numpy.
        """
        if model is None:
            model = self.policy_net
        model.eval()
        with torch.no_grad():
            x = self._prepare_input(board)
            q = model(x)
        return q.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Interface used by utils.play_game2()
    # ------------------------------------------------------------------

    def move(self, board, legal_moves, values=None):
        """
        Choose greedy action for each game (no epsilon here).
        utils.play_game2 does epsilon-greedy outside this function.

        board: [B, H, W, C]
        legal_moves: [B, n_actions] binary mask (1 = legal, 0 = illegal)
        """
        q_values = self._get_model_outputs(board, self.policy_net)  # [B, n_actions]
        # mask illegal moves
        masked_q = np.where(legal_moves == 1, q_values, -np.inf)
        actions = np.argmax(masked_q, axis=1)
        return actions

    def get_action_proba(self, board, values=None):
        """
        Convert Q-values to softmax probabilities (like original agent).
        """
        q_values = self._get_model_outputs(board, self.policy_net)
        q_values = np.clip(q_values, -10, 10)
        q_values = q_values - q_values.max(axis=1, keepdims=True)
        exp_q = np.exp(q_values)
        probs = exp_q / exp_q.sum(axis=1, keepdims=True)
        return probs

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        """
        Proxy to ReplayBufferNumpy.add_to_buffer()
        """
        self._buffer.add_to_buffer(board, action, reward, next_board, done, legal_moves)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_agent(self, batch_size=64, num_games=1, reward_clip=True):
        """
        Sample from replay buffer and perform one gradient step.

        Returns: float loss value
        """
        if self._buffer.get_current_size() < batch_size:
            return 0.0

        s, a_onehot, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        # shapes:
        #   s        : [B, H, W, C]
        #   a_onehot : [B, n_actions]   (0/1)
        #   r        : [B, 1]
        #   next_s   : [B, H, W, C]
        #   done     : [B, 1]
        #   legal    : [B, n_actions]

        if reward_clip:
            r = np.sign(r)

        # Convert everything to torch
        s_t = self._prepare_input(s)                  # [B, C, H, W]
        next_s_t = self._prepare_input(next_s)        # [B, C, H, W]

        actions = np.argmax(a_onehot, axis=1).astype(np.int64)
        actions_t = torch.from_numpy(actions).long().to(self.device)        # [B]
        rewards_t = torch.from_numpy(r).float().to(self.device).view(-1, 1) # [B,1]
        done_t = torch.from_numpy(done).float().to(self.device).view(-1, 1) # [B,1]
        legal_moves_t = torch.from_numpy(legal_moves.astype(np.bool_)).to(self.device)

        # Current Q values for chosen actions
        self.policy_net.train()
        q_values = self.policy_net(s_t)  # [B, n_actions]
        q_selected = q_values.gather(1, actions_t.view(-1, 1))  # [B,1]

        # Target Q values
        with torch.no_grad():
            if self._use_target_net:
                next_q_all = self.target_net(next_s_t)  # [B, n_actions]
            else:
                next_q_all = self.policy_net(next_s_t)

            # Mask illegal moves
            neg_inf = torch.tensor(-1e9, device=self.device)
            masked_next_q = torch.where(legal_moves_t, next_q_all, neg_inf)
            max_next_q, _ = masked_next_q.max(dim=1, keepdim=True)  # [B,1]

            target_q = rewards_t + self._gamma * max_next_q * (1.0 - done_t)

        loss = self.loss_fn(q_selected, target_q)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target_net(self):
        if self._use_target_net:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ------------------------------------------------------------------
    # Saving / loading (models + buffer)
    # ------------------------------------------------------------------

    def save_model(self, file_path: str = "", iteration: int | None = None):
        if iteration is None:
            iteration = 0
        os.makedirs(file_path, exist_ok=True)
        torch.save(
            self.policy_net.state_dict(),
            os.path.join(file_path, f"model_{iteration:04d}.pth"),
        )
        if self._use_target_net:
            torch.save(
                self.target_net.state_dict(),
                os.path.join(file_path, f"model_{iteration:04d}_target.pth"),
            )

    def load_model(self, file_path: str = "", iteration: int | None = None):
        if iteration is None:
            iteration = 0
        policy_path = os.path.join(file_path, f"model_{iteration:04d}.pth")
        target_path = os.path.join(file_path, f"model_{iteration:04d}_target.pth")
        if os.path.exists(policy_path):
            self.policy_net.load_state_dict(
                torch.load(policy_path, map_location=self.device)
            )
        if self._use_target_net and os.path.exists(target_path):
            self.target_net.load_state_dict(
                torch.load(target_path, map_location=self.device)
            )

    def save_buffer(self, file_path: str = "", iteration: int | None = None):
        if iteration is None:
            iteration = 0
        os.makedirs(file_path, exist_ok=True)
        with open(os.path.join(file_path, f"buffer_{iteration:04d}"), "wb") as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path: str = "", iteration: int | None = None):
        if iteration is None:
            iteration = 0
        buf_path = os.path.join(file_path, f"buffer_{iteration:04d}")
        if not os.path.exists(buf_path):
            raise FileNotFoundError(buf_path)
        with open(buf_path, "rb") as f:
            self._buffer = pickle.load(f)

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------

    def print_models(self):
        print(self.policy_net)
        if self._use_target_net:
            print("Target network:")
            print(self.target_net)
