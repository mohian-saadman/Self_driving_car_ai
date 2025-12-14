#!/usr/bin/env python3
"""
Self-Driving Car Racing Game
Main entry point for training and testing AI drivers
"""

import argparse
import torch
import pygame
from ai.trainer import Trainer
from game.environment import RacingEnvironment
from game.menu import Menu, RaceResults


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Self-Driving Car Racing Game')

    parser.add_argument(
        '--mode',
        type=str,
        default='menu',
        choices=['train', 'test', 'manual', 'race', 'menu'],
        help='Mode to run: train, test, manual, race, or menu'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=1000,
        help='Number of episodes for training (default: 1000)'
    )

    return parser.parse_args()


def get_device(device_arg):
    """Determine the best device to use."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_arg


def train_mode(args, model_name='new_model'):
    """Run training mode."""
    device = get_device('auto')
    trainer = Trainer(render=True, device=device)
    trainer.train(num_episodes=args.episodes, model_name=model_name)


def test_mode(model_path, num_episodes=10):
    """Run testing mode."""
    device = get_device('auto')
    trainer = Trainer(render=True, device=device)
    trainer.test(model_path=model_path, num_episodes=num_episodes)


def manual_mode():
    """Run manual control mode."""
    env = RacingEnvironment(render_mode=True, num_cars=1)
    running = True

    while running:
        env.reset()
        done = False
        while not done:
            actions = env.get_manual_action()
            _, _, dones, _ = env.step(actions)
            if dones:
                done = dones[0]

            if env.render():
                running = False
                break
    env.close()
    return


def race_mode(model_paths, num_cars):
    """Run race mode with multiple AI drivers and show results."""
    device = get_device('auto')
    trainer = Trainer(render=True, device=device)
    race_results = trainer.race(
        model_paths=model_paths, num_cars=num_cars, num_races=1)

    # Show results if available
    if race_results:
        print("\n" + "=" * 60)
        print("RACE RESULTS")
        print("=" * 60)
        standings = race_results[0]
        for position, car_index in enumerate(standings, 1):
            model_name = model_paths[car_index].replace(
                'models/', '').replace('.pth', '')
            print(f"{position}. Car {car_index + 1} - {model_name}")
        print("=" * 60 + "\n")

    return race_results


def menu_mode():
    """Display the main menu."""
    pygame.init()
    # Start with default size, but allow dynamic resizing
    screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
    pygame.display.set_caption("Main Menu")
    menu = Menu(screen)
    menu_result = menu.run()
    pygame.quit()
    return menu_result


def main():
    """Main entry point."""
    args = parse_args()

    while True:
        menu_result = menu_mode()

        selected_mode = menu_result.get('mode')

        print("=" * 60)
        print("Self-Driving Car Racing Game")
        print("=" * 60)

        if selected_mode == 'train':
            print("Mode: Training")
            model_name = menu_result.get('model_name', 'new_model')
            train_mode(args, model_name=model_name)
        elif selected_mode == 'test':
            print("Mode: Testing")
            model_path = menu_result.get('model', 'models/best_model.pth')
            test_mode(model_path)
        elif selected_mode == 'manual':
            print("Mode: Manual")
            manual_mode()
        elif selected_mode == 'race':
            print("Mode: Racing")
            num_cars = menu_result.get('num_cars', 2)
            model_paths = menu_result.get(
                'race_models', ['models/best_model.pth'] * num_cars)
            race_results = race_mode(model_paths, num_cars)

            # Show results screen
            if race_results:
                pygame.init()
                results_screen = pygame.display.set_mode(
                    (800, 600), pygame.RESIZABLE)
                pygame.display.set_caption("Race Results")
                results_display = RaceResults(
                    results_screen, race_results[0], model_paths)
                results_display.run()
                pygame.quit()
        elif selected_mode == 'quit' or selected_mode is None:
            break

    print("\nProgram finished!")


if __name__ == '__main__':
    main()
