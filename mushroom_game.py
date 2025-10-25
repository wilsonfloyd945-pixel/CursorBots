"""Мини-игра о гриб-человеке, вдохновлённая "Превращением" Кафки.

Запустить игру можно командой ``python mushroom_game.py``. Управление —
стрелками или WASD. Чтобы выйти, нажмите ESC или закройте окно.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pygame


# ---------------------------------------------------------------------------
# Базовые цвета и размеры помещения
# ---------------------------------------------------------------------------

WIDTH = 900
HEIGHT = 600

WALL_COLOR = (214, 205, 182)
FLOOR_COLOR = (178, 150, 120)
BED_COLOR = (160, 172, 189)
BED_FRAME_COLOR = (110, 94, 82)
DOOR_FRAME_COLOR = (92, 71, 55)
DOOR_VIEW_COLOR = (52, 42, 40)
TEXT_COLOR = (40, 35, 30)
HIGHLIGHT_COLOR = (230, 214, 170)


# ---------------------------------------------------------------------------
# Вспомогательные структуры
# ---------------------------------------------------------------------------


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_lines(text: str, font: pygame.font.Font, max_width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current: List[str] = []

    for word in words:
        test_line = " ".join(current + [word]) if current else word
        if font.size(test_line)[0] <= max_width:
            current.append(word)
            continue

        if current:
            lines.append(" ".join(current))
            current = [word]
        else:
            # слово целиком не влезает — разрезаем по символам
            split: List[str] = []
            for char in word:
                test_line = "".join(split + [char])
                if font.size(test_line)[0] <= max_width:
                    split.append(char)
                else:
                    lines.append("".join(split))
                    split = [char]
            if split:
                current = ["".join(split)]

    if current:
        lines.append(" ".join(current))

    return lines


@dataclass
class DialogueLine:
    author: str
    text: str
    ttl: float


class FamilyMember:
    """Представление члена семьи, живущего за дверью."""

    def __init__(
        self,
        name: str,
        color: Tuple[int, int, int],
        base_x: float,
        lines: Iterable[str],
    ) -> None:
        self.name = name
        self.color = color
        self.base_x = base_x
        self.lines = list(lines)
        self._phase = random.random() * math.tau
        self._speak_timer = random.uniform(2.0, 6.0)

    def update(self, dt: float) -> Optional[DialogueLine]:
        self._phase = (self._phase + dt * 1.5) % math.tau
        self._speak_timer -= dt

        if self._speak_timer <= 0:
            self._speak_timer = random.uniform(6.0, 12.0)
            text = random.choice(self.lines)
            return DialogueLine(self.name, text, 5.0)

        return None

    def position(self, door_rect: pygame.Rect) -> Tuple[float, float]:
        amplitude = door_rect.height * 0.18
        x = door_rect.left + self.base_x * door_rect.width
        y = door_rect.centery + math.sin(self._phase) * amplitude
        return x, y

    def draw(self, surface: pygame.Surface, door_rect: pygame.Rect) -> None:
        x, y = self.position(door_rect)
        radius = door_rect.width * 0.18
        pygame.draw.circle(surface, self.color, (int(x), int(y)), int(radius))
        pygame.draw.rect(
            surface,
            self.color,
            pygame.Rect(
                int(x - radius * 0.6),
                int(y),
                int(radius * 1.2),
                int(radius * 1.7),
            ),
            border_radius=int(radius * 0.4),
        )


class MushroomPlayer:
    """Игрок-гриб, проснувшийся в собственной комнате."""

    def __init__(self, position: Tuple[float, float]) -> None:
        self.position = pygame.Vector2(position)
        self.speed = 150.0

    def update(self, dt: float, keys: pygame.key.ScancodeWrapper, room_rect: pygame.Rect) -> None:
        direction = pygame.Vector2(0, 0)

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            direction.x -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            direction.x += 1
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            direction.y -= 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            direction.y += 1

        if direction.length_squared() > 0:
            direction = direction.normalize()

        self.position += direction * self.speed * dt
        self.position.x = clamp(self.position.x, room_rect.left + 40, room_rect.right - 40)
        self.position.y = clamp(self.position.y, room_rect.top + 50, room_rect.bottom - 35)

    def draw(self, surface: pygame.Surface) -> None:
        stem_width = 26
        stem_height = 70
        cap_width = 80
        cap_height = 45

        stem_rect = pygame.Rect(0, 0, stem_width, stem_height)
        stem_rect.center = (int(self.position.x), int(self.position.y))

        cap_rect = pygame.Rect(0, 0, cap_width, cap_height)
        cap_rect.center = (int(self.position.x), int(self.position.y) - stem_height // 2)

        pygame.draw.rect(surface, (232, 222, 202), stem_rect, border_radius=12)
        pygame.draw.ellipse(surface, (196, 60, 72), cap_rect)
        pygame.draw.ellipse(surface, (240, 229, 213), cap_rect.inflate(-cap_width * 0.4, -cap_height * 0.5))


class MushroomGame:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Грегор-гриб")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("arial", 22)
        self.small_font = pygame.font.SysFont("arial", 18)

        room_margin = 80
        self.room_rect = pygame.Rect(
            room_margin,
            room_margin,
            WIDTH - room_margin * 2,
            HEIGHT - room_margin * 2,
        )
        self.door_rect = pygame.Rect(
            self.room_rect.right - 64,
            self.room_rect.top + 120,
            46,
            180,
        )

        self.player = MushroomPlayer((self.room_rect.centerx - 120, self.room_rect.bottom - 120))

        self.family: List[FamilyMember] = [
            FamilyMember(
                "Сестра",
                (210, 138, 138),
                0.3,
                (
                    "Мне нужно на репетицию, но как оставить Грегора одного?",
                    "Если бы он просто открыл дверь…",
                    "Он всегда так много работал…",
                ),
            ),
            FamilyMember(
                "Мать",
                (203, 176, 138),
                0.55,
                (
                    "Ты слышал, как он шевелился?",
                    "Может быть, позвать врача?",
                    "Вчера он почти не ел…",
                ),
            ),
            FamilyMember(
                "Отец",
                (130, 130, 140),
                0.8,
                (
                    "Без него мы не сможем платить ренту.",
                    "Работа не ждёт, а он там заперся.",
                    "Никто не войдёт, пока он сам не попросит.",
                ),
            ),
        ]

        self.dialogue_queue: List[DialogueLine] = []
        self.active_line: Optional[DialogueLine] = None
        self.active_timer = 0.0

        self.running = True

    # ------------------------------------------------------------------
    # Игровой цикл
    # ------------------------------------------------------------------

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            self.handle_events()
            self.update(dt)
            self.draw()

        pygame.quit()

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False

    def update(self, dt: float) -> None:
        keys = pygame.key.get_pressed()
        self.player.update(dt, keys, self.room_rect)

        for member in self.family:
            line = member.update(dt)
            if line is not None:
                self.dialogue_queue.append(line)

        if self.active_line is None and self.dialogue_queue:
            self.active_line = self.dialogue_queue.pop(0)
            self.active_timer = 0.0

        if self.active_line is not None:
            self.active_timer += dt
            if self.active_timer >= self.active_line.ttl:
                self.active_line = None

    # ------------------------------------------------------------------
    # Отрисовка сцены
    # ------------------------------------------------------------------

    def draw(self) -> None:
        self.draw_room()
        self.draw_door_view()
        self.player.draw(self.screen)
        self.draw_ui()
        pygame.display.flip()

    def draw_room(self) -> None:
        self.screen.fill(WALL_COLOR)

        # лёгкий узор на стенах
        wallpaper = pygame.Surface((WIDTH, HEIGHT))
        wallpaper.set_alpha(28)
        for y in range(0, HEIGHT, 48):
            for x in range(0, WIDTH, 48):
                rect = pygame.Rect(x, y, 24, 24)
                pygame.draw.rect(wallpaper, (255, 255, 255), rect, width=1)
        self.screen.blit(wallpaper, (0, 0))

        pygame.draw.rect(self.screen, FLOOR_COLOR, self.room_rect)

        # простая кровать
        bed_rect = pygame.Rect(
            self.room_rect.left + 60,
            self.room_rect.centery - 40,
            200,
            120,
        )
        pygame.draw.rect(self.screen, BED_FRAME_COLOR, bed_rect.inflate(16, 16))
        pygame.draw.rect(self.screen, BED_COLOR, bed_rect, border_radius=14)
        pillow_rect = pygame.Rect(0, 0, 80, 40)
        pillow_rect.midright = (bed_rect.right - 16, bed_rect.top + 35)
        pygame.draw.rect(self.screen, (230, 230, 240), pillow_rect, border_radius=12)

        # окно
        window_rect = pygame.Rect(
            self.room_rect.left + 40,
            self.room_rect.top + 30,
            170,
            110,
        )
        pygame.draw.rect(self.screen, (160, 190, 215), window_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), window_rect, width=6)
        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (window_rect.centerx, window_rect.top),
            (window_rect.centerx, window_rect.bottom),
            width=4,
        )
        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (window_rect.left, window_rect.centery),
            (window_rect.right, window_rect.centery),
            width=4,
        )

        # стол
        desk_rect = pygame.Rect(
            self.room_rect.centerx + 70,
            self.room_rect.top + 80,
            160,
            80,
        )
        pygame.draw.rect(self.screen, (120, 96, 82), desk_rect)
        pygame.draw.rect(
            self.screen,
            (100, 82, 70),
            pygame.Rect(desk_rect.x + 20, desk_rect.bottom, 20, 80),
        )
        pygame.draw.rect(
            self.screen,
            (100, 82, 70),
            pygame.Rect(desk_rect.right - 40, desk_rect.bottom, 20, 80),
        )

        pygame.draw.rect(self.screen, DOOR_FRAME_COLOR, self.door_rect.inflate(18, 18))

    def draw_door_view(self) -> None:
        interior = pygame.Surface(self.door_rect.size)
        interior.fill(DOOR_VIEW_COLOR)
        pygame.draw.rect(
            interior,
            (78, 64, 58),
            pygame.Rect(0, 0, self.door_rect.width, self.door_rect.height),
            width=6,
        )

        for member in self.family:
            member.draw(interior, interior.get_rect())

        self.screen.blit(interior, self.door_rect.topleft)

    def draw_ui(self) -> None:
        instructions = (
            "Вы — Грегор Замза, ныне гриб.\n"
            "Подойдите к двери, чтобы прислушаться к жизни семьи."
        )
        self.draw_text_block(instructions, (self.room_rect.left + 16, self.room_rect.bottom + 12), width=380)

        if self.active_line is not None:
            text = f"{self.active_line.author}: {self.active_line.text}"
            text_rect = pygame.Rect(
                self.room_rect.left + 20,
                self.room_rect.top - 60,
                self.room_rect.width - 40,
                52,
            )
            self.draw_text_box(text, text_rect)

        # мысли Грегора возле двери
        distance = pygame.Vector2(self.door_rect.center).distance_to(self.player.position)
        if distance < 170:
            thoughts = (
                "За дверью — обычная жизнь.\n"
                "Но как снова стать сыном и братом,\n"
                "когда ты — гриб?"
            )
            box_rect = pygame.Rect(
                self.door_rect.left - 280,
                self.door_rect.top - 120,
                260,
                110,
            )
            self.draw_text_box(thoughts, box_rect, align="center")

    # ------------------------------------------------------------------
    # Рисование текста
    # ------------------------------------------------------------------

    def draw_text_block(self, text: str, topleft: Tuple[int, int], width: int) -> None:
        lines = wrap_lines(text, self.small_font, width)
        x, y = topleft
        for line in lines:
            surface = self.small_font.render(line, True, TEXT_COLOR)
            self.screen.blit(surface, (x, y))
            y += self.small_font.get_linesize()

    def draw_text_box(
        self,
        text: str,
        rect: pygame.Rect,
        *,
        align: str = "left",
        padding: int = 12,
    ) -> None:
        pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, rect, border_radius=12)
        pygame.draw.rect(self.screen, (120, 106, 88), rect, width=2, border_radius=12)

        inner_rect = rect.inflate(-padding * 2, -padding * 2)
        lines = wrap_lines(text, self.font, inner_rect.width)

        y = inner_rect.top
        for line in lines:
            surface = self.font.render(line, True, TEXT_COLOR)
            if align == "center":
                x = inner_rect.centerx - surface.get_width() // 2
            else:
                x = inner_rect.left
            self.screen.blit(surface, (x, y))
            y += self.font.get_linesize()


def main() -> None:
    game = MushroomGame()
    game.run()


if __name__ == "__main__":
    main()

