import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_progress(rewards, avg_rewards, save_path="training_progress.png"):
    """Plot training rewards over episodes."""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, label='Episode Reward')
    plt.plot(avg_rewards, label='Average Reward (100 episodes)', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    print(f"Training progress plot saved to {save_path}")


def plot_episode_lengths(episode_lengths, save_path="episode_lengths.png"):
    """Plot episode lengths over training."""
    plt.figure(figsize=(10, 6))
    plt.plot(episode_lengths, alpha=0.6)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episode Lengths')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    print(f"Episode lengths plot saved to {save_path}")


def create_directories():
    """Create necessary directories for saving models and plots."""
    directories = ['models', 'plots']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def print_training_stats(episode, reward, avg_reward, epsilon, episode_length):
    """Print formatted training statistics."""
    print(f"Episode {episode:4d} | "
          f"Reward: {reward:7.2f} | "
          f"Avg Reward: {avg_reward:7.2f} | "
          f"Epsilon: {epsilon:.3f} | "
          f"Steps: {episode_length:4d}")


def save_best_model_info(episode, reward, save_path="models/best_model_info.txt"):
    """Save information about the best model."""
    with open(save_path, 'w') as f:
        f.write(f"Best Model Information\n")
        f.write(f"=====================\n")
        f.write(f"Episode: {episode}\n")
        f.write(f"Reward: {reward:.2f}\n")
    print(f"Best model info saved to {save_path}")


class MovingAverage:
    """Calculate moving average of values."""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.values = []

    def add(self, value):
        """Add a new value."""
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def get(self):
        """Get current moving average."""
        if not self.values:
            return 0
        return np.mean(self.values)
