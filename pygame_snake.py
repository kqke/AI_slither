import pygame
import sys
from constants import *
from pygame.locals import *


BLOCK_SIZE = 20
SCORE_BOARD = 200
BORDER = 20

TITLE_X_OFFSET = 40
TITLE_Y_OFFSET = 5
SCORE_X_OFFSET = TITLE_X_OFFSET + 5
SCORE_Y_OFFSET = 100
SCORE_X_SPACE = 2
SCORE_Y_SPACE = SCORE_FONT_SZ

HEAD_X_OFFSET = 5
HEAD_Y_OFFSET = 5


def play_gui(game, turns):
    i = 0
    s = init_screen(game)
    clock = pygame.time.Clock()
    while i < turns:
        clock.tick(10)
        for e in pygame.event.get():
            if e.type == QUIT:
                sys.exit(0)
        draw_board(s, game)
        draw_scores(s, game)
        pygame.display.update()
        game.play_turn()
        i += 1
    # todo
    # display round results or something
    # display_results(s, game)
    sys.exit(0)


def init_screen(game):
    pygame.init()
    pygame.font.init()
    w = game.get_width() * BLOCK_SIZE + SCORE_BOARD + (2 * BORDER)
    h = game.get_height() * BLOCK_SIZE + (2 * BORDER)
    s = pygame.display.set_mode((w, h))
    pygame.display.set_caption(SLITHER)
    s.fill(BLACK)
    x = game.get_width() * BLOCK_SIZE + BORDER + TITLE_X_OFFSET
    y = BORDER + TITLE_Y_OFFSET
    draw_logo(s, (x, y))
    return s


def draw_logo(screen, loc):
    snake_font = pygame.font.Font(SNAKE_FONT, SNAKE_FONT_SZ)
    text = snake_font.render(SLITHER, False, GREEN)
    screen.blit(text, loc)


def draw_board(screen, game):
    h = game.get_height()
    w = game.get_width()
    state = game.get_state()
    for y in range(h):
        for x in range(w):
            rect = pygame.Rect(BORDER + x * BLOCK_SIZE, BORDER + y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            if state[y, x] == 100:
                pygame.draw.rect(screen, GREEN, rect)
            elif state[y, x] == 0:
                pygame.draw.rect(screen, WHITE, rect)
            elif state[y, x] < 0:
                pid = int(abs(state[y, x]))
                draw_head(screen, game.get_player(pid), rect)
            else:
                p_type = game.get_player_type(int(abs(state[y, x])))
                if p_type == RANDOM_PLAYER:
                    colour = random_color()
                else:
                    colour = SNAKE_COLORS[p_type]
                pygame.draw.rect(screen, colour, rect)


def draw_head(screen, player, rect):
    direction = player.get_direction()
    pid = str(player.get_id())
    score_font = pygame.font.Font(SCORE_FONT, PID_FONT_SZ)
    text = score_font.render(pid, False, WHITE)

    if direction == LEFT or direction == RIGHT:
        text = pygame.transform.rotate(text, 90)
    elif direction == DOWN:
        text = pygame.transform.rotate(text, 180)

    x = rect.x + HEAD_X_OFFSET
    y = rect.y + HEAD_Y_OFFSET

    p_type = player.get_type()

    if p_type == RANDOM_PLAYER:
        colour = random_color()
    else:
        colour = SNAKE_COLORS[p_type]

    pygame.draw.rect(screen, colour, rect)
    screen.blit(text, (x, y))


def draw_scores(screen, game):
    top = BORDER + SCORE_Y_OFFSET
    left = game.get_width() * BLOCK_SIZE + BORDER
    w, h = pygame.display.get_surface().get_size()
    rect = pygame.Rect(left, top, w - left, h - top)
    pygame.draw.rect(screen, BLACK, rect)
    x = game.get_width() * BLOCK_SIZE + BORDER + SCORE_X_OFFSET
    y = BORDER + SCORE_Y_OFFSET
    score_font = pygame.font.Font(SCORE_FONT, SCORE_FONT_SZ)
    for player in game.get_players():
        score = '%d %s' % (player.get_id(), player.get_type()) + ' ' * SCORE_X_SPACE + '%d' % player.get_score()
        text = score_font.render(score, False, WHITE)
        screen.blit(text, (x, y))
        y += SCORE_Y_SPACE
