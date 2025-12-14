import pygame
import math
import numpy as np


class Track:
    """Represents the racing track with boundaries and checkpoints."""

    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2

        # Oval track parameters
        self.outer_radius_x = 300
        self.outer_radius_y = 200
        self.inner_radius_x = 200
        self.inner_radius_y = 100
        self.track_width = 100

        # Generate track boundaries
        self.outer_points = self._generate_oval(
            self.outer_radius_x, self.outer_radius_y)
        self.inner_points = self._generate_oval(
            self.inner_radius_x, self.inner_radius_y)

        # Checkpoints for lap tracking
        self.checkpoints = self._generate_checkpoints(16)
        self.start_line = (self.center_x + self.outer_radius_x, self.center_y)

    def _generate_oval(self, radius_x, radius_y, num_points=100):
        """Generate points for an oval shape."""
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = self.center_x + radius_x * math.cos(angle)
            y = self.center_y + radius_y * math.sin(angle)
            points.append((x, y))
        return points

    def _generate_checkpoints(self, num_checkpoints):
        """Generate checkpoint lines across the track."""
        checkpoints = []
        for i in range(num_checkpoints):
            angle = 2 * math.pi * i / num_checkpoints

            # Inner point
            inner_x = self.center_x + self.inner_radius_x * math.cos(angle)
            inner_y = self.center_y + self.inner_radius_y * math.sin(angle)

            # Outer point
            outer_x = self.center_x + self.outer_radius_x * math.cos(angle)
            outer_y = self.center_y + self.outer_radius_y * math.sin(angle)

            checkpoints.append({
                'id': i,
                'inner': (inner_x, inner_y),
                'outer': (outer_x, outer_y),
                'angle': angle
            })
        return checkpoints

    def is_on_track(self, x, y):
        """Check if a point is on the track (between inner and outer boundaries)."""
        dx = x - self.center_x
        dy = y - self.center_y

        # Normalize to oval space
        dist_outer = (dx / self.outer_radius_x) ** 2 + \
            (dy / self.outer_radius_y) ** 2
        dist_inner = (dx / self.inner_radius_x) ** 2 + \
            (dy / self.inner_radius_y) ** 2

        return dist_inner >= 1.0 and dist_outer <= 1.0

    def get_distance_to_center_line(self, x, y):
        """Get perpendicular distance to track center line."""
        dx = x - self.center_x
        dy = y - self.center_y

        # Average radius for center line
        center_radius_x = (self.outer_radius_x + self.inner_radius_x) / 2
        center_radius_y = (self.outer_radius_y + self.inner_radius_y) / 2

        angle = math.atan2(dy, dx)
        center_x = self.center_x + center_radius_x * math.cos(angle)
        center_y = self.center_y + center_radius_y * math.sin(angle)

        return math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    def render(self, screen):
        """Render the track on the screen."""
        # New Color Scheme
        bg_color = (18, 32, 47)  # Dark Blue
        track_color = (52, 73, 94)  # Wet Asphalt
        line_color = (236, 239, 241)  # Off-white
        start_line_color = (231, 76, 60)  # Pomegranate

        # Draw background
        screen.fill(bg_color)

        # Draw outer boundary (track color)
        pygame.draw.polygon(screen, track_color, self.outer_points, 0)

        # Draw inner boundary (background color)
        pygame.draw.polygon(screen, bg_color, self.inner_points, 0)

        # Draw track lines
        pygame.draw.lines(screen, line_color, True, self.outer_points, 2)
        pygame.draw.lines(screen, line_color, True, self.inner_points, 2)

        # Draw start/finish line
        start_inner = (self.center_x + self.inner_radius_x, self.center_y)
        start_outer = (self.center_x + self.outer_radius_x, self.center_y)
        pygame.draw.line(screen, start_line_color, start_inner, start_outer, 4)

        # Draw checkpoints (for debugging)
        # for cp in self.checkpoints:
        #     pygame.draw.line(screen, (255, 255, 0), cp['inner'], cp['outer'], 1)
