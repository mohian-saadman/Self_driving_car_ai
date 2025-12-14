import numpy as np
import pygame
from game.track import Track
from game.car import Car


class Button:
    def __init__(self, x, y, width, height, text, color, hover_color, text_color=(255, 255, 255)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font = pygame.font.Font(None, 30)

    def draw(self, screen, mouse_pos):
        is_hovering = self.rect.collidepoint(mouse_pos)
        color = self.hover_color if is_hovering else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)


class RacingEnvironment:
    """Gym-like environment for the racing game."""

    def __init__(self, render_mode=True, num_cars=1):
        self.render_mode = render_mode
        self.num_cars = num_cars

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("Self-Driving Car Racing")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 28)
            self.stop_button = Button(
                680, 550, 100, 40, "Stop", (200, 50, 50), (250, 80, 80))

        self.track = Track(800, 600)
        self.cars = self._create_cars()
        self.action_space_n = 7
        self.observation_space_n = 8
        self.episode_steps = 0
        self.max_episode_steps = 2000

    def _create_cars(self):
        cars = []
        colors = [
            pygame.Color("#1abc9c"), pygame.Color("#3498db"),
            pygame.Color("#9b59b6"), pygame.Color("#f1c40f"),
            pygame.Color("#e74c3c")
        ]
        # Place cars along the start line with vertical offsets so they all start on-track
        spacing = 30
        for i in range(self.num_cars):
            start_x = self.track.center_x + self.track.outer_radius_x - 50
            # center the group vertically around the track center
            offset = (i - (self.num_cars - 1) / 2) * spacing
            start_y = self.track.center_y + offset
            car = Car(start_x, start_y, angle=0, color=colors[i % len(colors)])
            cars.append(car)
        return cars

    def reset(self):
        self.episode_steps = 0
        # Reset cars to the start line with the same vertical spacing used at creation
        spacing = 30
        for i, car in enumerate(self.cars):
            start_x = self.track.center_x + self.track.outer_radius_x - 50
            offset = (i - (self.num_cars - 1) / 2) * spacing
            start_y = self.track.center_y + offset
            car.reset(start_x, start_y, angle=0)
        return [car.get_state() for car in self.cars]

    def step(self, actions):
        self.episode_steps += 1
        next_states, rewards, dones = [], [], []
        for car, action in zip(self.cars, actions):
            if car.alive:
                prev_checkpoint_count = len(car.checkpoints_passed)
                car.update(action, self.track)
                for checkpoint in self.track.checkpoints:
                    car.check_checkpoint(
                        checkpoint, len(self.track.checkpoints))
                reward = self._calculate_reward(car, prev_checkpoint_count)
                done = self._is_done(car)
            else:
                reward, done = 0, True
            next_states.append(car.get_state())
            rewards.append(reward)
            dones.append(done)
        return next_states, rewards, dones, {}

    def _calculate_reward(self, car, prev_checkpoint_count):
        reward = - \
            100 if not car.alive else (
                len(car.checkpoints_passed) - prev_checkpoint_count) * 50
        if car.alive:
            reward += car.speed * 0.1 - (0.5 if car.speed < 1.0 else 0)
            reward -= self.track.get_distance_to_center_line(
                car.x, car.y) * 0.01 + 0.1
        return reward

    def _is_done(self, car):
        return not car.alive or self.episode_steps >= self.max_episode_steps or len(car.checkpoints_passed) >= len(self.track.checkpoints)

    def render(self):
        if not self.render_mode:
            return False

        should_quit = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                should_quit = True
            if self.stop_button.is_clicked(event):
                return True  # Signal to stop

        self.track.render(self.screen)
        for car in self.cars:
            car.render(self.screen)

        y_offset = 10
        for i, car in enumerate(self.cars):
            info = f"Car {i+1}: Speed: {car.speed:.1f}, Checkpoints: {len(car.checkpoints_passed)}"
            text_surface = self.font.render(info, True, car.color)
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 30

        self.stop_button.draw(self.screen, pygame.mouse.get_pos())
        pygame.display.flip()
        self.clock.tick(60)
        return should_quit

    def check_stop_button(self, event):
        if self.render_mode:
            return self.stop_button.is_clicked(event)
        return False

    def close(self):
        if self.render_mode:
            pygame.quit()

    def get_manual_action(self):
        keys = pygame.key.get_pressed()
        action = 0
        if keys[pygame.K_UP]:
            action = 1
        elif keys[pygame.K_DOWN]:
            action = 2
        elif keys[pygame.K_LEFT]:
            action = 3
        elif keys[pygame.K_RIGHT]:
            action = 4
        if keys[pygame.K_UP] and keys[pygame.K_LEFT]:
            action = 5
        if keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
            action = 6
        return [action] * self.num_cars
