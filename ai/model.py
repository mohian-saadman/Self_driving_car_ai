import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQN(nn.Module):
    """Deep Q-Network for the racing car."""

    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for training the racing car."""

    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.target_update_frequency = 10

        # Networks
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and loss
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Replay buffer
        self.memory = ReplayBuffer(capacity=10000)

        # Training statistics
        self.training_step = 0

    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(
                state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q_values = self.policy_net(
            states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + \
                (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update training statistics
        self.training_step += 1

        return loss.item()

    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save model to file."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load model from file."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
        except Exception as e:
            # Newer PyTorch versions (2.6+) default to weights_only=True which
            # can raise UnpicklingError for older checkpoints that include
            # non-tensor objects. Retry with weights_only=False as a fallback.
            try:
                checkpoint = torch.load(
                    filepath, map_location=self.device, weights_only=False)
                import warnings
                warnings.warn(
                    f"Loaded checkpoint {filepath} with weights_only=False fallback."
                )
            except TypeError:
                # weights_only not supported in this torch version; re-raise original
                raise e
        # Normalize multiple possible checkpoint formats:
        # - Full dict with keys like 'policy_net_state_dict' (agent.save)
        # - Trainer checkpoints with keys like 'policy_net_state'
        # - A raw state_dict (weights-only) saved directly
        # - Other naming variants

        # If checkpoint looks like a state_dict for a single network (keys contain '.')
        if isinstance(checkpoint, dict):
            # Detect if dict is a state_dict (param tensors) by checking keys
            sample_keys = list(checkpoint.keys())[:10]
            looks_like_state_dict = any('.' in k for k in sample_keys) and all(
                torch.is_tensor(v) or isinstance(v, (torch.Tensor,)) for v in list(checkpoint.values())[:10]
            ) if sample_keys else False

            if looks_like_state_dict:
                # Treat as policy_net state_dict
                self.policy_net.load_state_dict(checkpoint)
                self.target_net.load_state_dict(self.policy_net.state_dict())
                print(f"Model (weights-only) loaded from {filepath}")
                return

            # Helper to find a key from candidates
            def find_key(candidates):
                for c in candidates:
                    if c in checkpoint:
                        return c
                return None

            policy_key = find_key(
                ['policy_net_state_dict', 'policy_net_state', 'policy_net_state_dicts', 'policy_net'])
            target_key = find_key(
                ['target_net_state_dict', 'target_net_state', 'target_net'])
            optim_key = find_key(
                ['optimizer_state_dict', 'optimizer_state', 'optimizer_state_dicts', 'optimizer'])
            eps_key = find_key(['epsilon'])
            step_key = find_key(
                ['training_step', 'training_steps', 'step', 'episode'])

            # If keys not found, try common nested keys
            if policy_key is None and 'state_dict' in checkpoint:
                # some checkpoints store under 'state_dict'
                nested = checkpoint['state_dict']
                if isinstance(nested, dict):
                    # assume nested is policy state_dict
                    self.policy_net.load_state_dict(nested)
                    self.target_net.load_state_dict(
                        self.policy_net.state_dict())
                    print(
                        f"Model loaded from {filepath} (nested 'state_dict')")
                    return

            # Load if we found explicit keys
            if policy_key is not None:
                self.policy_net.load_state_dict(checkpoint[policy_key])
            else:
                raise KeyError(
                    f"No policy network state found in checkpoint: {filepath}")

            if target_key is not None:
                try:
                    self.target_net.load_state_dict(checkpoint[target_key])
                except Exception:
                    # fallback: copy from policy
                    self.target_net.load_state_dict(
                        self.policy_net.state_dict())
            else:
                # if no target key, copy policy weights
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if optim_key is not None:
                try:
                    self.optimizer.load_state_dict(checkpoint[optim_key])
                except Exception:
                    # optimizer state may be incompatible across devices/versions
                    pass

            if eps_key is not None:
                try:
                    self.epsilon = checkpoint[eps_key]
                except Exception:
                    pass

            if step_key is not None:
                try:
                    self.training_step = checkpoint[step_key]
                except Exception:
                    pass

            print(f"Model loaded from {filepath}")
            return

        else:
            # Unknown checkpoint type
            raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")
