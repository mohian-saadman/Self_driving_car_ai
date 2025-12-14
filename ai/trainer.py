import torch
import numpy as np
import os
from game.environment import RacingEnvironment
from ai.model import DQNAgent
from game.utils import (
    plot_training_progress,
    plot_episode_lengths,
    print_training_stats,
    save_best_model_info,
    MovingAverage,
    create_directories
)


class Trainer:
    """Trainer for the DQN agent."""

    def __init__(self, render=False, device='cpu'):
        self.device = device
        self.render = render
        self.env = None
        self.agent = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.avg_rewards = []
        self.best_reward = float('-inf')
        self.moving_avg = MovingAverage(window_size=100)
        create_directories()

    def _init_single_agent_env(self):
        """Initialize environment and agent for single-car modes."""
        self.env = RacingEnvironment(render_mode=self.render, num_cars=1)
        self.agent = DQNAgent(
            state_size=self.env.observation_space_n,
            action_size=self.env.action_space_n,
            device=self.device
        )

    def train(self, num_episodes=1000, save_frequency=50, model_name='new_model'):
        """Train the agent with checkpoint saving and resume capability."""
        self._init_single_agent_env()

        # Check if checkpoint exists to resume training
        checkpoint_path = f"models/checkpoint_{model_name}_latest.pth"
        start_episode = 1

        if os.path.exists(checkpoint_path):
            print(f"Found checkpoint at {checkpoint_path}")
            response = input(
                "Resume training from checkpoint? (y/n): ").lower()
            if response == 'y':
                checkpoint = torch.load(checkpoint_path)
                self.agent.policy_net.load_state_dict(
                    checkpoint['policy_net_state'])
                self.agent.target_net.load_state_dict(
                    checkpoint['target_net_state'])
                self.agent.optimizer.load_state_dict(
                    checkpoint['optimizer_state'])
                self.episode_rewards = checkpoint['episode_rewards']
                self.episode_lengths = checkpoint['episode_lengths']
                self.avg_rewards = checkpoint['avg_rewards']
                self.best_reward = checkpoint['best_reward']
                start_episode = checkpoint['episode'] + 1
                print(f"Resumed from episode {start_episode}")

        print(f"Starting training for model: {model_name}")
        print(f"Episodes: {start_episode} to {num_episodes}")
        print("=" * 70)

        stopped = False
        for episode in range(start_episode, num_episodes + 1):
            if stopped:
                break

            states = self.env.reset()
            state = states[0]
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action = self.agent.select_action(state, training=True)
                next_states, rewards, dones, _ = self.env.step([action])
                next_state, reward, done = next_states[0], rewards[0], dones[0]

                self.agent.store_transition(
                    state, action, reward, next_state, done)
                self.agent.train_step()
                state = next_state
                episode_reward += reward
                episode_length += 1

                if self.render and self.env.render():
                    stopped = True
                    break

            # Track episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.moving_avg.add(episode_reward)
            avg_reward = self.moving_avg.get()
            self.avg_rewards.append(avg_reward)

            # Print stats every 10 episodes
            if episode % 10 == 0:
                print_training_stats(
                    episode, episode_reward, avg_reward, self.agent.epsilon, episode_length)

            # Save resume checkpoint after every episode (overwrites previous)
            checkpoint_data = {
                'episode': episode,
                'policy_net_state': self.agent.policy_net.state_dict(),
                'target_net_state': self.agent.target_net.state_dict(),
                'optimizer_state': self.agent.optimizer.state_dict(),
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'avg_rewards': self.avg_rewards,
                'best_reward': self.best_reward,
            }
            torch.save(checkpoint_data, checkpoint_path)

            # Save best model
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                best_path = f"models/{model_name}_best.pth"
                self.agent.save(best_path)
                save_best_model_info(
                    episode, episode_reward, f"models/{model_name}_best_info.txt")
                print(
                    f"*** NEW BEST MODEL: Episode {episode} with reward {episode_reward:.2f} ***")

        # Training complete
        print("=" * 70)
        print(
            f"Training completed! Total episodes: {len(self.episode_rewards)}")

        # Save final model
        final_path = f"models/{model_name}_final.pth"
        self.agent.save(final_path)
        print(f"Final model saved to {final_path}")

        # Generate plots
        progress_plot = f"plots/{model_name}_training_progress.png"
        lengths_plot = f"plots/{model_name}_episode_lengths.png"
        plot_training_progress(self.episode_rewards,
                               self.avg_rewards, progress_plot)
        plot_episode_lengths(self.episode_lengths, lengths_plot)

        self.env.close()

    def test(self, model_path, num_episodes=10):
        """Test a trained model."""
        self._init_single_agent_env()
        print(f"Testing model: {model_path}")
        self.agent.load(model_path)
        self.agent.epsilon = 0

        for episode in range(1, num_episodes + 1):
            states = self.env.reset()
            state = states[0]
            done = False
            while not done:
                action = self.agent.select_action(state, training=False)
                next_states, rewards, dones, _ = self.env.step([action])
                state, _, done = next_states[0], rewards[0], dones[0]

                if self.render and self.env.render():
                    self.env.close()
                    return
        self.env.close()

    def race(self, model_paths, num_cars, num_races=1):
        """Race multiple trained models and return results."""
        print(f"Starting race with {num_cars} cars...")
        self.env = RacingEnvironment(
            render_mode=self.render, num_cars=num_cars)

        agents = []
        for i in range(num_cars):
            model_path = model_paths[i % len(model_paths)]
            agent = DQNAgent(
                state_size=self.env.observation_space_n,
                action_size=self.env.action_space_n,
                device=self.device
            )
            agent.load(model_path)
            agents.append(agent)

        race_results = []

        for race_num in range(1, num_races + 1):
            states = self.env.reset()
            all_done = False

            # Track finish step for each car (None until finished)
            finished_at = [None] * num_cars

            while not all_done:
                actions = [agents[i].select_action(
                    states[i], training=False) for i in range(num_cars)]
                states, rewards, dones, _ = self.env.step(actions)

                # Record finish step when a car completes all checkpoints (finish),
                # do not treat crashes (alive=False) as a finish.
                for i in range(num_cars):
                    try:
                        cp_count = len(
                            getattr(self.env.cars[i], 'checkpoints_passed', []))
                        total_cp = len(
                            getattr(self.env.track, 'checkpoints', []))
                    except Exception:
                        cp_count = 0
                        total_cp = 0
                    is_finished = (total_cp > 0 and cp_count >= total_cp)
                    if is_finished and finished_at[i] is None:
                        # Use environment's episode_steps as the finish time
                        finished_at[i] = getattr(
                            self.env, 'episode_steps', None)

                # If user requested to stop (via render stop button), compute partial standings
                if self.render and self.env.render():
                    # For cars not finished, use negative progress (more checkpoints higher progress)
                    progress = [
                        len(getattr(self.env.cars[i], 'checkpoints_passed', [])) for i in range(num_cars)]
                    # Create a tuple (finished_at or large, -progress) so higher progress ranks ahead
                    ranking_key = [((finished_at[i] if finished_at[i] is not None else float(
                        'inf')), -progress[i]) for i in range(num_cars)]
                    standings = sorted(
                        range(num_cars), key=lambda i: ranking_key[i])
                    race_results.append(standings)
                    self.env.close()
                    return race_results

                # Stop the race when any car completes all checkpoints (first to finish)
                any_finished = any(
                    (len(getattr(self.env.cars[i], 'checkpoints_passed', [])) >= len(
                        getattr(self.env.track, 'checkpoints', [])))
                    for i in range(num_cars)
                )
                if any_finished:
                    # Ensure finished_at recorded for finished cars
                    for i in range(num_cars):
                        cp_count = len(
                            getattr(self.env.cars[i], 'checkpoints_passed', []))
                        total_cp = len(
                            getattr(self.env.track, 'checkpoints', []))
                        if total_cp > 0 and cp_count >= total_cp and finished_at[i] is None:
                            finished_at[i] = getattr(
                                self.env, 'episode_steps', None)
                    all_done = True

            # Determine standings - earlier finish (lower finished_at) wins; tie-breaker by progress
            progress = [len(getattr(self.env.cars[i], 'checkpoints_passed', []))
                        for i in range(num_cars)]
            ranking_key = [((finished_at[i] if finished_at[i] is not None else float(
                'inf')), -progress[i]) for i in range(num_cars)]
            standings = sorted(range(num_cars), key=lambda i: ranking_key[i])
            race_results.append(standings)

        self.env.close()
        return race_results
