import pygame
import sys
import random
from constants import *
from config import *
from pygame.locals import *


def play_gui(game, turns):
    i = 0
    s = init_screen(game)
    clock = pygame.time.Clock()
    while i < turns:
        clock.tick(GUI_DELAY)
        for e in pygame.event.get():
            if e.type == QUIT:
                pygame.quit()
                quit()
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    pause(s)
                elif e.key == pygame.K_p:
                    capture(s)
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
    w = GAME_WIDTH * BLOCK_SIZE + SCORE_BOARD + (2 * BORDER)
    h = GAME_HEIGHT * BLOCK_SIZE + (2 * BORDER)
    s = pygame.display.set_mode((w, h))
    pygame.display.set_caption(SLITHER)
    s.fill(BLACK)
    x = GAME_WIDTH * BLOCK_SIZE + BORDER + TITLE_X_OFFSET
    y = BORDER + TITLE_Y_OFFSET
    draw_logo(s, (x, y))
    draw_names(s, game)
    return s


def draw_logo(screen, loc):
    snake_font = pygame.font.Font(SNAKE_FONT, SNAKE_FONT_SZ)
    text = snake_font.render(SLITHER, False, GREEN)
    screen.blit(text, loc)


def draw_names(screen, game):
    x = GAME_WIDTH * BLOCK_SIZE + BORDER + SCORE_X_OFFSET
    y = BORDER + SCORE_Y_OFFSET
    score_font = pygame.font.Font(SCORE_FONT, SCORE_FONT_SZ)
    for player in game.get_players():
        score = '%d %s' % (player.get_id(), player.get_type())
        text = score_font.render(score, False, WHITE)
        screen.blit(text, (x, y))
        y += SCORE_Y_SPACE


def pause(screen):
    # draw_pause(screen)
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    return
                elif e.key == pygame.K_p:
                    capture(screen)


# todo
# finish
# maybe add some indicator that a screen shot was taken
def capture(screen):
    rect = pygame.Rect(BORDER, BORDER, GAME_WIDTH * BLOCK_SIZE, GAME_HEIGHT * BLOCK_SIZE)
    sub = screen.subsurface(rect)
    pygame.image.save(sub, SCREEN_SHOT_DIR + "/screen_shot.jpg")


# doesn't work...
# probably not necessary
# it's possible to do something nice here though
def draw_pause(screen):
    x = (GAME_WIDTH - 2) * BLOCK_SIZE
    y = (GAME_HEIGHT - 2) * BLOCK_SIZE
    # x = GAME_CENTER_X
    # y = GAME_CENTER_Y
    b_d = BLOCK_SIZE // 3
    rect_1 = pygame.Rect(x, y, b_d, BLOCK_SIZE)
    rect_2 = pygame.Rect(x + 2 * b_d, y, b_d, BLOCK_SIZE)
    pygame.draw.rect(screen, BLACK, rect_1)
    pygame.draw.rect(screen, BLACK, rect_2)


def draw_board(screen, game):
    h = GAME_HEIGHT
    w = GAME_WIDTH
    state = game.get_state()
    for y in range(h):
        for x in range(w):
            rect = pygame.Rect(BORDER + x * BLOCK_SIZE, BORDER + y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            if state[y, x] == FOOD_MARK:
                pygame.draw.rect(screen, GREEN, rect)
            elif state[y, x] == FREE_SQUARE_MARK:
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


def random_color():
    rgb = ()
    for _ in range(3):
        rgb += (random.randint(0, 255), )
    return rgb


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
    w, h = pygame.display.get_surface().get_size()
    players = game.get_players()
    y = BORDER + SCORE_Y_OFFSET
    score_font = pygame.font.Font(SCORE_FONT, SCORE_FONT_SZ)
    for player in players:
        score = str(player.get_score())
        width = (len(score) + 1) * SCORE_FONT_W
        x = w - (BORDER + width)
        rect = pygame.Rect(x, y, BORDER + width, SCORE_Y_SPACE)
        pygame.draw.rect(screen, BLACK, rect)
        text = score_font.render(score, False, WHITE)
        screen.blit(text, (x, y))
        y += SCORE_Y_SPACE
