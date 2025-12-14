import pygame
import pygame
import sys
import os

# UI Theme
THEME = {
    'bg': (12, 22, 34),
    'panel': (28, 40, 54),
    'accent': (88, 172, 255),
    'accent_dark': (48, 112, 200),
    'muted': (150, 160, 170),
    'success': (46, 204, 113),
    'danger': (231, 76, 60),
    'white': (236, 239, 241),
    'shadow': (0, 0, 0, 60)
}


class Button:
    def __init__(self, x, y, width, height, text, color, hover_color, text_color=(255, 255, 255), radius=10):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font = pygame.font.Font(None, 34)
        self.radius = radius

    def draw(self, screen, mouse_pos):
        is_hovering = self.rect.collidepoint(mouse_pos)
        color = self.hover_color if is_hovering else self.color

        # shadow
        shadow_rect = self.rect.copy()
        shadow_rect.x += 3
        shadow_rect.y += 3
        s = pygame.Surface(
            (shadow_rect.width, shadow_rect.height), pygame.SRCALPHA)
        s.fill((0, 0, 0, 90))
        screen.blit(s, (shadow_rect.x, shadow_rect.y))

        # main button
        pygame.draw.rect(screen, color, self.rect, border_radius=self.radius)
        pygame.draw.rect(screen, tuple(min(255, c + 30)
                         for c in color), self.rect, 3, border_radius=self.radius)
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)


class ModelSelector:
    """Simple model selector using arrow buttons instead of dropdowns.

    Responsive: accepts a width and positions arrow buttons relative to the box.
    """

    def __init__(self, x, y, model_files, width=300, label=""):
        self.rect = pygame.Rect(x, y, width, 50)
        self.model_files = model_files
        self.current_index = 0
        self.selected_model = model_files[0] if model_files else ""
        self.label = label
        self.font = pygame.font.Font(None, 28)
        self.label_font = pygame.font.Font(None, 24)

        # Left and right arrow buttons (position relative to rect)
        self.left_btn = pygame.Rect(x - 60, y, 50, 50)
        self.right_btn = pygame.Rect(x + width + 10, y, 50, 50)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.left_btn.collidepoint(event.pos):
                self.current_index = (
                    self.current_index - 1) % len(self.model_files)
                self.selected_model = self.model_files[self.current_index]
            elif self.right_btn.collidepoint(event.pos):
                self.current_index = (
                    self.current_index + 1) % len(self.model_files)
                self.selected_model = self.model_files[self.current_index]

    def draw(self, screen, mouse_pos):
        # Draw label
        if self.label:
            label_surf = self.label_font.render(
                self.label, True, THEME['white'])
            screen.blit(label_surf, (self.rect.x, self.rect.y - 30))

        # Draw main selector box
        pygame.draw.rect(screen, THEME['panel'], self.rect, border_radius=8)
        pygame.draw.rect(screen, THEME['accent_dark'],
                         self.rect, 2, border_radius=8)

        # Draw selected model name (truncate with ellipsis)
        model_text = self.selected_model
        max_chars = max(12, int(self.rect.width / 12))
        if len(model_text) > max_chars:
            model_text = model_text[:max_chars - 3] + '...'
        text_surf = self.font.render(model_text, True, THEME['white'])
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

        # Draw left arrow button
        left_hover = self.left_btn.collidepoint(mouse_pos)
        pygame.draw.rect(
            screen, THEME['accent'] if left_hover else THEME['accent_dark'], self.left_btn, border_radius=6)
        pygame.draw.polygon(screen, THEME['white'], [
            (self.left_btn.x + 28, self.left_btn.y + 15),
            (self.left_btn.x + 18, self.left_btn.y + 25),
            (self.left_btn.x + 28, self.left_btn.y + 35)
        ])

        # Draw right arrow button
        right_hover = self.right_btn.collidepoint(mouse_pos)
        pygame.draw.rect(
            screen, THEME['accent'] if right_hover else THEME['accent_dark'], self.right_btn, border_radius=6)
        pygame.draw.polygon(screen, THEME['white'], [
            (self.right_btn.x + 22, self.right_btn.y + 15),
            (self.right_btn.x + 32, self.right_btn.y + 25),
            (self.right_btn.x + 22, self.right_btn.y + 35)
        ])

        # Draw counter
        counter_text = f"{self.current_index + 1}/{len(self.model_files)}"
        counter_surf = pygame.font.Font(None, 20).render(
            counter_text, True, THEME['muted'])
        screen.blit(counter_surf, (self.rect.x +
                    self.rect.width - 60, self.rect.y + 55))


class TextInput:
    def __init__(self, x, y, width, height, placeholder="Enter text"):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = ""
        self.placeholder = placeholder
        self.font = pygame.font.Font(None, 32)
        self.is_focused = False
        self.cursor_visible = True
        self.cursor_timer = 0

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.is_focused = self.rect.collidepoint(event.pos)
        elif event.type == pygame.KEYDOWN and self.is_focused:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                self.text += event.unicode

    def draw(self, screen):
        # Background
        bg = THEME['panel'] if not self.is_focused else tuple(
            min(255, c + 8) for c in THEME['panel'])
        pygame.draw.rect(screen, bg, self.rect, border_radius=8)
        pygame.draw.rect(
            screen, THEME['accent_dark'] if self.is_focused else THEME['bg'], self.rect, 2, border_radius=8)

        display_text = self.text if self.text else self.placeholder
        text_color = THEME['white'] if self.text else THEME['muted']
        text_surf = self.font.render(display_text, True, text_color)
        screen.blit(text_surf, (self.rect.x + 12, self.rect.y + 10))

        if self.is_focused and self.cursor_visible:
            cursor_x = self.rect.x + 12 + text_surf.get_width()
            pygame.draw.line(screen, THEME['white'], (cursor_x, self.rect.y + 8),
                             (cursor_x, self.rect.y + self.rect.height - 8), 2)


class SearchableDropdown:
    def __init__(self, x, y, width, height, options, label=""):
        self.rect = pygame.Rect(x, y, width, height)
        self.options = options
        self.filtered_options = options
        self.selected_option = options[0] if options else ""
        self.font = pygame.font.Font(None, 28)
        self.label_font = pygame.font.Font(None, 24)
        self.is_open = False
        self.search_text = ""
        self.scroll_offset = 0
        self.max_visible_options = 3
        self.label = label
        self.hovered_option = None

    def handle_event(self, event, mouse_pos):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.is_open = not self.is_open
                self.search_text = ""
                self.filtered_options = self.options
            elif self.is_open:
                for i, option in enumerate(self.filtered_options[self.scroll_offset:self.scroll_offset + self.max_visible_options]):
                    rect = self.rect.copy()
                    rect.y += (i + 1) * self.rect.height
                    if rect.collidepoint(event.pos):
                        self.selected_option = option
                        self.is_open = False
                        break
                else:
                    self.is_open = False

        if event.type == pygame.KEYDOWN and self.is_open:
            if event.key == pygame.K_BACKSPACE:
                self.search_text = self.search_text[:-1]
            elif event.key == pygame.K_ESCAPE:
                self.is_open = False
            else:
                self.search_text += event.unicode
            self.filtered_options = [
                opt for opt in self.options if self.search_text.lower() in opt.lower()]
            self.scroll_offset = 0

        if event.type == pygame.MOUSEWHEEL and self.is_open:
            self.scroll_offset -= event.y
            if self.scroll_offset < 0:
                self.scroll_offset = 0
            if self.scroll_offset > max(0, len(self.filtered_options) - self.max_visible_options):
                self.scroll_offset = max(
                    0, len(self.filtered_options) - self.max_visible_options)

        # Track hovered option for mouse movement
        if self.is_open:
            for i, option in enumerate(self.filtered_options[self.scroll_offset:self.scroll_offset + self.max_visible_options]):
                rect = self.rect.copy()
                rect.y += (i + 1) * self.rect.height
                if rect.collidepoint(mouse_pos):
                    self.hovered_option = i
                    return
        self.hovered_option = None

    def draw(self, screen, mouse_pos):
        # Draw label if provided
        if self.label:
            label_surf = pygame.font.Font(None, 24).render(
                self.label, True, (236, 239, 241))
            screen.blit(label_surf, (self.rect.x, self.rect.y - 30))

        # Draw main dropdown box
        is_hovering = self.rect.collidepoint(mouse_pos)
        color = THEME['panel'] if not self.is_open else tuple(
            min(255, c + 6) for c in THEME['panel'])
        border_color = THEME['accent_dark'] if self.is_open else THEME['bg']

        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, border_color, self.rect, 3, border_radius=8)

        text_surf = self.font.render(
            self.selected_option[:30], True, THEME['white'])
        screen.blit(text_surf, (self.rect.x + 12, self.rect.y + 10))

        # Draw dropdown arrow
        arrow_x = self.rect.x + self.rect.width - 20
        arrow_y = self.rect.y + self.rect.height // 2
        if self.is_open:
            pygame.draw.polygon(screen, (255, 255, 255), [
                (arrow_x - 5, arrow_y + 3),
                (arrow_x + 5, arrow_y + 3),
                (arrow_x, arrow_y - 3)
            ])
        else:
            pygame.draw.polygon(screen, (255, 255, 255), [
                (arrow_x - 5, arrow_y - 3),
                (arrow_x + 5, arrow_y - 3),
                (arrow_x, arrow_y + 3)
            ])

        # Draw dropdown options
        if self.is_open:
            for i, option in enumerate(self.filtered_options[self.scroll_offset:self.scroll_offset + self.max_visible_options]):
                rect = self.rect.copy()
                rect.y += (i + 1) * self.rect.height

                # Highlight on hover
                is_hovered = rect.collidepoint(mouse_pos)
                option_color = tuple(
                    min(255, c + 10) for c in (THEME['accent'] if is_hovered else THEME['panel']))
                pygame.draw.rect(screen, option_color, rect, border_radius=6)
                pygame.draw.rect(
                    screen, THEME['accent_dark'] if is_hovered else THEME['bg'], rect, 2, border_radius=6)

                text_surf = self.font.render(option[:40], True, THEME['white'])
                screen.blit(text_surf, (rect.x + 12, rect.y + 10))

            # Draw scrollbar indicator if needed
            if len(self.filtered_options) > self.max_visible_options:
                scroll_rect = self.rect.copy()
                scroll_rect.x += self.rect.width - 5
                scroll_rect.y += self.rect.height
                scroll_rect.height = self.max_visible_options * self.rect.height
                scroll_rect.width = 4
                pygame.draw.rect(screen, (150, 150, 150), scroll_rect)

                # Draw scroll position indicator
                scroll_pos = self.rect.copy()
                scroll_pos.x += self.rect.width - 5
                scroll_pos.y += self.rect.height + (self.scroll_offset / max(1, len(
                    self.filtered_options) - self.max_visible_options)) * (self.max_visible_options - 1) * self.rect.height
                scroll_pos.height = self.rect.height
                scroll_pos.width = 4
                pygame.draw.rect(screen, (200, 200, 200), scroll_pos)


class RaceResults:
    """Display race results screen with scrolling support"""

    def __init__(self, screen, standings, model_paths):
        self.screen = screen
        self.width, self.height = screen.get_size()
        self.standings = standings
        self.model_paths = model_paths
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.Font(None, 72)
        self.font_large = pygame.font.Font(None, 48)
        self.font_normal = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        self.running = True
        self.bg_color = (18, 32, 47)
        self.title_color = (236, 239, 241)
        self.button_color = (52, 73, 94)
        self.button_hover_color = (44, 62, 80)
        self.continue_button = Button(
            self.width // 2 - 100, self.height - 80, 200, 50, "Continue", self.button_color, self.button_hover_color)

        # Scrolling variables
        self.scroll_offset = 0
        # Approximate content height
        self.content_height = 180 + len(standings) * 100 + 100
        self.max_scroll = max(0, self.content_height - (self.height - 200))

    def run(self):
        """Display results until user clicks continue"""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if self.continue_button.is_clicked(event):
                    self.running = False
                # Handle scrolling with mouse wheel
                if event.type == pygame.MOUSEWHEEL:
                    self.scroll_offset -= event.y * 30
                    self.scroll_offset = max(
                        0, min(self.scroll_offset, self.max_scroll))
                # Handle scrolling with arrow keys
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.scroll_offset -= 30
                        self.scroll_offset = max(0, self.scroll_offset)
                    elif event.key == pygame.K_DOWN:
                        self.scroll_offset += 30
                        self.scroll_offset = min(
                            self.scroll_offset, self.max_scroll)

            self.draw()
            pygame.display.flip()
            self.clock.tick(60)

    def draw(self):
        """Draw the results screen"""
        # Background
        self.screen.fill(THEME['bg'])

        # Title bar
        title_bar = pygame.Rect(0, 0, self.width, 96)
        pygame.draw.rect(self.screen, THEME['panel'], title_bar)
        title_surf = self.font_title.render(
            "Race Results", True, THEME['white'])
        self.screen.blit(title_surf, title_surf.get_rect(
            center=(self.width // 2, 48)))

        # Create scrollable area
        scrollable_rect = pygame.Rect(
            40, 120, self.width - 80, self.height - 200)

        # Draw standings as cards with elevation
        y_offset = 140 - self.scroll_offset
        for position, car_index in enumerate(self.standings):
            card_h = 84
            card_rect = pygame.Rect(60, y_offset, self.width - 140, card_h)
            # background card
            pygame.draw.rect(
                self.screen, THEME['panel'], card_rect, border_radius=8)
            # accent for winner
            if position == 0:
                pygame.draw.rect(
                    self.screen, THEME['accent'], (card_rect.x, card_rect.y, 8, card_h), border_radius=4)

            # Texts
            model_path = self.model_paths[car_index]
            model_name = model_path.replace('models/', '').replace('.pth', '')
            pos_label = f"#{position + 1}"
            pos_surf = self.font_large.render(pos_label, True, THEME['white'])
            self.screen.blit(pos_surf, (card_rect.x + 20, card_rect.y + 12))

            car_label = f"Car {car_index + 1}"
            car_surf = self.font_normal.render(car_label, True, THEME['white'])
            self.screen.blit(car_surf, (card_rect.x + 120, card_rect.y + 8))

            model_surf = self.font_small.render(
                model_name[:60], True, THEME['muted'])
            self.screen.blit(model_surf, (card_rect.x + 120, card_rect.y + 40))

            y_offset += card_h + 16

        # Draw scrollbar if needed
        if self.max_scroll > 0:
            scrollbar_rect = pygame.Rect(
                self.width - 20, 120, 15, self.height - 200)
            pygame.draw.rect(self.screen, (50, 50, 50), scrollbar_rect)

            scroll_position = (self.scroll_offset /
                               self.max_scroll) * (self.height - 200 - 50)
            scroll_thumb = pygame.Rect(
                self.width - 20, 120 + scroll_position, 15, 50)
            pygame.draw.rect(self.screen, (150, 150, 150), scroll_thumb)

        # Continue button (stays at bottom)
        self.continue_button.draw(self.screen, pygame.mouse.get_pos())


class Menu:
    def __init__(self, screen):
        self.screen = screen
        self.width, self.height = screen.get_size()
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.Font(None, 72)
        self.font_small = pygame.font.Font(None, 28)
        self.running = True
        self.current_menu = "main"
        self.menu_result = None

        self.bg_color = (18, 32, 47)
        self.title_color = (236, 239, 241)
        self.button_color = (52, 73, 94)
        self.button_hover_color = (44, 62, 80)

        self.buttons = {
            "main": {"manual": Button(self.width // 2 - 150, 200, 300, 50, "Manual Mode", self.button_color, self.button_hover_color), "train": Button(self.width // 2 - 150, 270, 300, 50, "Train New Model", self.button_color, self.button_hover_color), "test": Button(self.width // 2 - 150, 340, 300, 50, "Test a Model", self.button_color, self.button_hover_color), "race": Button(self.width // 2 - 150, 410, 300, 50, "Race Models", self.button_color, self.button_hover_color), },
            "train": {"start": Button(self.width // 2 - 150, 350, 300, 50, "Start Training", self.button_color, self.button_hover_color), "back": Button(self.width // 2 - 150, 420, 300, 50, "Back", self.button_color, self.button_hover_color), },
            "test": {"start": Button(self.width // 2 - 150, 420, 300, 50, "Start Test", self.button_color, self.button_hover_color), "back": Button(self.width // 2 - 150, 490, 300, 50, "Back", self.button_color, self.button_hover_color), },
            "race": {"start": Button(self.width // 2 - 150, 500, 300, 50, "Start Race", self.button_color, self.button_hover_color), "back": Button(self.width // 2 - 150, 560, 300, 50, "Back", self.button_color, self.button_hover_color), "add": Button(self.width // 2 + 100, 150, 50, 50, "+", self.button_color, self.button_hover_color), "remove": Button(self.width // 2 - 150, 150, 50, 50, "-", self.button_color, self.button_hover_color), }
        }
        self.model_files = sorted(
            [f for f in os.listdir("models") if f.endswith(".pth")])
        self.test_model_selector = ModelSelector(
            self.width // 2 - 150, 250, self.model_files, label="Select Model")

        self.num_cars_race = 2
        self.race_model_selectors = []
        self._update_race_selectors()
        self.train_model_name_input = TextInput(
            self.width // 2 - 150, 250, 300, 50, placeholder="Enter model name")

    def _update_race_selectors(self):
        """Update race model selectors with proper spacing based on window size."""
        # Preserve current selections
        prev_selected = [s.selected_model for s in self.race_model_selectors]
        self.race_model_selectors = []

        # Responsive layout: arrange selectors in columns/rows to keep them visible
        top_margin = 180
        bottom_margin = 120
        selector_h = 50
        available_h = max(100, self.height - top_margin - bottom_margin)
        max_rows = max(1, available_h // (selector_h + 30))
        cols = (self.num_cars_race + max_rows - 1) // max_rows
        col_width = min(420, max(280, (self.width - 100) // max(1, cols)))

        for i in range(self.num_cars_race):
            col = i // max_rows
            row = i % max_rows
            x = 50 + col * col_width
            y = top_margin + row * (selector_h + 30)
            selector = ModelSelector(
                x, y, self.model_files, width=col_width - 40, label=f"Car {i+1} Model")

            # Restore previous selection if present
            if i < len(prev_selected):
                prev = prev_selected[i]
                if prev in self.model_files:
                    selector.current_index = self.model_files.index(prev)
                    selector.selected_model = prev

            self.race_model_selectors.append(selector)

    def run(self):
        while self.running:
            self.handle_events()

            # Handle window resizing
            if self.current_menu == "race":
                new_width, new_height = self.screen.get_size()
                if new_width != self.width or new_height != self.height:
                    self.width, self.height = new_width, new_height
                    self._update_race_selectors()

            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        return self.menu_result

    def draw(self):
        self.screen.fill(self.bg_color)
        if self.current_menu == "main":
            self.main_menu()
        elif self.current_menu == "train":
            self.train_menu()
        elif self.current_menu == "test":
            self.test_menu()
        elif self.current_menu == "race":
            self.race_menu()

    def main_menu(self):
        title_surf = self.font_title.render(
            "Self-Driving Car", True, self.title_color)
        self.screen.blit(title_surf, title_surf.get_rect(
            center=(self.width // 2, 100)))
        for button in self.buttons["main"].values():
            button.draw(self.screen, pygame.mouse.get_pos())

    def train_menu(self):
        title_surf = self.font_title.render(
            "Train New Model", True, self.title_color)
        self.screen.blit(title_surf, title_surf.get_rect(
            center=(self.width // 2, 100)))

        label_surf = self.font_small.render(
            "Model Name:", True, self.title_color)
        self.screen.blit(label_surf, (self.width // 2 - 150, 200))

        self.train_model_name_input.draw(self.screen)
        for button in self.buttons["train"].values():
            button.draw(self.screen, pygame.mouse.get_pos())

    def test_menu(self):
        title_surf = self.font_title.render(
            "Test Model", True, self.title_color)
        self.screen.blit(title_surf, title_surf.get_rect(
            center=(self.width // 2, 100)))
        self.test_model_selector.draw(self.screen, pygame.mouse.get_pos())

        for button in self.buttons["test"].values():
            button.draw(self.screen, pygame.mouse.get_pos())

    def race_menu(self):
        title_surf = self.font_title.render(
            "Race Setup", True, self.title_color)
        self.screen.blit(title_surf, title_surf.get_rect(
            center=(self.width // 2, 100)))
        num_cars_text = self.font_small.render(
            f"Number of Cars: {self.num_cars_race}", True, self.title_color)
        self.screen.blit(num_cars_text, num_cars_text.get_rect(
            center=(self.width // 2, 150)))

        mouse_pos = pygame.mouse.get_pos()

        # Draw selectors
        for selector in self.race_model_selectors:
            selector.draw(self.screen, mouse_pos)

        # Position control buttons (+ and -) near the selectors area
        # place them at top-left of selector region
        selector_area_x = 50
        self.buttons["race"]["add"].rect.x = selector_area_x + 200
        self.buttons["race"]["add"].rect.y = 120
        self.buttons["race"]["remove"].rect.x = selector_area_x - 60
        self.buttons["race"]["remove"].rect.y = 120

        # Position start and back buttons pinned to footer (always visible)
        footer_y = self.height - 70
        gap = 20
        total_width = self.buttons["race"]["start"].rect.width + \
            gap + self.buttons["race"]["back"].rect.width
        self.buttons["race"]["start"].rect.x = self.width // 2 - \
            total_width // 2
        self.buttons["race"]["start"].rect.y = footer_y
        self.buttons["race"]["back"].rect.x = self.buttons["race"]["start"].rect.x + \
            self.buttons["race"]["start"].rect.width + gap
        self.buttons["race"]["back"].rect.y = footer_y

        for button in self.buttons["race"].values():
            button.draw(self.screen, mouse_pos)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.menu_result = {"mode": "quit"}

            if self.current_menu == "main":
                if self.buttons["main"]["train"].is_clicked(event):
                    self.current_menu = "train"
                elif self.buttons["main"]["test"].is_clicked(event):
                    self.current_menu = "test"
                elif self.buttons["main"]["race"].is_clicked(event):
                    self.current_menu = "race"
                elif self.buttons["main"]["manual"].is_clicked(event):
                    self.running = False
                    self.menu_result = {"mode": "manual"}
            elif self.current_menu == "train":
                self.train_model_name_input.handle_event(event)
                if self.buttons["train"]["start"].is_clicked(event):
                    self.running = False
                    model_name = self.train_model_name_input.text if self.train_model_name_input.text else "new_model"
                    self.menu_result = {
                        "mode": "train", "model_name": model_name}
                elif self.buttons["train"]["back"].is_clicked(event):
                    self.current_menu = "main"
            elif self.current_menu == "test":
                self.test_model_selector.handle_event(event)

                if self.buttons["test"]["start"].is_clicked(event):
                    self.running = False
                    selected = self.test_model_selector.selected_model
                    if os.path.isabs(selected):
                        model_path = selected
                    else:
                        model_path = f"models/{selected}"
                    self.menu_result = {"mode": "test", "model": model_path}
                elif self.buttons["test"]["back"].is_clicked(event):
                    self.current_menu = "main"
            elif self.current_menu == "race":
                for selector in self.race_model_selectors:
                    selector.handle_event(event)
                if self.buttons["race"]["add"].is_clicked(event) and self.num_cars_race < 5:
                    self.num_cars_race += 1
                    self._update_race_selectors()
                elif self.buttons["race"]["remove"].is_clicked(event) and self.num_cars_race > 2:
                    self.num_cars_race -= 1
                    self._update_race_selectors()
                elif self.buttons["race"]["start"].is_clicked(event):
                    self.running = False
                    models = [
                        f"models/{sel.selected_model}" for sel in self.race_model_selectors]
                    self.menu_result = {
                        "mode": "race", "num_cars": self.num_cars_race, "race_models": models}
                elif self.buttons["race"]["back"].is_clicked(event):
                    self.current_menu = "main"


if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    menu = Menu(screen)
    print(menu.run())
    pygame.quit()
