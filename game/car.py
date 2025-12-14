import pygame
import math
import numpy as np


class Car:
    """Represents a racing car with physics and sensors."""

    def __init__(self, x, y, angle=0, color=(252, 186, 3)):
        self.x = x
        self.y = y
        self.angle = angle  # in radians
        self.speed = 0
        self.color = color

        # Car dimensions
        self.width = 20
        self.height = 40

        # Physics parameters
        self.max_speed = 8
        self.acceleration = 0.3
        self.friction = 0.95
        self.turn_speed = 0.08

        # Sensor parameters
        self.sensor_length = 150
        self.sensor_angles = [-60, -30, 0, 30, 60]  # degrees relative to car
        self.sensor_readings = [0] * len(self.sensor_angles)

        # State tracking
        self.alive = True
        self.checkpoints_passed = set()
        self.current_checkpoint = 0
        self.lap_time = 0
        self.laps_completed = 0

    def reset(self, x, y, angle=0):
        """Reset car to initial position."""
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0
        self.alive = True
        self.checkpoints_passed = set()
        self.current_checkpoint = 0
        self.lap_time = 0
        self.laps_completed = 0
        self.sensor_readings = [0] * len(self.sensor_angles)

    def update(self, action, track):
        """Update car state based on action.

        Actions:
        0: Do nothing
        1: Accelerate
        2: Brake
        3: Turn left
        4: Turn right
        5: Accelerate + Turn left
        6: Accelerate + Turn right
        """
        if not self.alive:
            return

        # Apply action
        if action in [1, 5, 6]:  # Accelerate
            self.speed += self.acceleration
        elif action == 2:  # Brake
            self.speed -= self.acceleration * 1.5

        if action in [3, 5]:  # Turn left
            self.angle -= self.turn_speed
        elif action in [4, 6]:  # Turn right
            self.angle += self.turn_speed

        # Apply friction
        self.speed *= self.friction

        # Limit speed
        self.speed = max(-self.max_speed / 2, min(self.speed, self.max_speed))

        # Update position
        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)

        # Update sensors
        self.update_sensors(track)

        # Check if still on track
        if not track.is_on_track(self.x, self.y):
            self.alive = False

        # Update lap time
        self.lap_time += 1

    def update_sensors(self, track):
        """Cast rays to detect track boundaries."""
        for i, angle_offset in enumerate(self.sensor_angles):
            angle_rad = self.angle + math.radians(angle_offset)

            # Cast ray
            distance = 0
            step = 5
            while distance < self.sensor_length:
                distance += step
                end_x = self.x + distance * math.cos(angle_rad)
                end_y = self.y + distance * math.sin(angle_rad)

                if not track.is_on_track(end_x, end_y):
                    break

            # Normalize distance (0 = max distance, 1 = collision)
            self.sensor_readings[i] = 1.0 - (distance / self.sensor_length)

    def check_checkpoint(self, checkpoint, num_checkpoints):
        """Check if car has crossed a checkpoint."""
        if checkpoint['id'] in self.checkpoints_passed:
            return False

        # Simple line intersection check
        cp_inner = checkpoint['inner']
        cp_outer = checkpoint['outer']

        # Check if car position is near checkpoint line
        dist_to_line = self._point_to_line_distance(
            (self.x, self.y), cp_inner, cp_outer
        )

        if dist_to_line < 20:  # threshold
            # Check if crossing in the right direction
            expected_checkpoint = self.current_checkpoint
            if checkpoint['id'] == expected_checkpoint:
                self.checkpoints_passed.add(checkpoint['id'])
                self.current_checkpoint = (
                    checkpoint['id'] + 1) % num_checkpoints
                return True

        return False

    def _point_to_line_distance(self, point, line_start, line_end):
        """Calculate perpendicular distance from point to line segment."""
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))

        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    def get_state(self):
        """Get current state for RL agent."""
        # Normalize values
        state = np.array([
            self.speed / self.max_speed,  # normalized speed
            math.cos(self.angle),  # angle components
            math.sin(self.angle),
            *self.sensor_readings,  # all sensor readings
        ], dtype=np.float32)
        return state

    def render(self, screen):
        """Render the car on screen."""
        if not self.alive:
            # Draw crashed car in gray
            color = (128, 128, 128)
        else:
            color = self.color

        # Calculate car corners
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)

        half_width = self.width / 2
        half_height = self.height / 2

        corners = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height)
        ]

        rotated_corners = []
        for cx, cy in corners:
            rx = cx * cos_a - cy * sin_a
            ry = cx * sin_a + cy * cos_a
            rotated_corners.append((self.x + rx, self.y + ry))

        # Draw car
        pygame.draw.polygon(screen, color, rotated_corners)

        # Draw direction indicator (front of car)
        front_x = self.x + half_height * cos_a
        front_y = self.y + half_height * sin_a
        pygame.draw.circle(screen, (255, 255, 255),
                           (int(front_x), int(front_y)), 3)

        # Draw sensors (for debugging)
        if self.alive:
            for i, angle_offset in enumerate(self.sensor_angles):
                angle_rad = self.angle + math.radians(angle_offset)
                distance = (1.0 - self.sensor_readings[i]) * self.sensor_length
                end_x = self.x + distance * math.cos(angle_rad)
                end_y = self.y + distance * math.sin(angle_rad)

                color = (0, 255, 0) if self.sensor_readings[i] < 0.5 else (
                    255, 255, 0)
                pygame.draw.line(screen, color, (int(self.x), int(self.y)),
                                 (int(end_x), int(end_y)), 1)
