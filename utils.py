import numpy as np

from constants import *
from config import *


DIRECTION_TO_N_ROT90 = {
    UP: 0,
    RIGHT: 1,
    DOWN: 2,
    LEFT: 3
}


# general utils
def l1_distance(p1, p2):
    y1, x1 = p1
    y2, x2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def sample_bool_matrix(mat):
    indices = np.argwhere(mat)
    n = indices.shape[0]
    i = np.random.randint(n)
    loc = tuple(indices[i].tolist())
    return loc


# greedy action util
def get_greedy_action(game, head, direction):
    food = game.get_food()
    if len(food) == 0:
        return FORWARD_ACTION

    # find the nearest food
    food_and_distances = []
    for food_xy in food:
        dist = l1_distance(food_xy, head)
        food_and_distances.append((food_xy, dist))
    min_food = min(food_and_distances, key=lambda fd: fd[1])[0]

    # find the action which get us closest the the nearest food
    state = game.get_state()
    actions_indices_and_distances = []
    for i, action in enumerate(ACTIONS):
        action_direction = convert_action_to_direction(action, direction)
        action_loc = get_next_location(head, action_direction)

        # avoid obvious collisions
        if state[action_loc] in [FOOD_MARK, FREE_SQUARE_MARK]:
            dist = l1_distance(min_food, action_loc)
        else:  # snake's body or another snake
            dist = float("inf")

        actions_indices_and_distances.append((action, dist))

    greedy_action = min(actions_indices_and_distances, key=lambda ad: ad[1])[0]
    return greedy_action


# game utils
def center_state(state, center):
    center_y, center_x = center
    centered_state = np.roll(np.roll(state, GAME_CENTER_Y - center_y, axis=0), GAME_CENTER_X - center_x, axis=1)
    return centered_state


def rotate_state(state, direction):
    n_rot90 = DIRECTION_TO_N_ROT90[direction]
    rotated_state = np.rot90(state, n_rot90)
    return rotated_state


def normalize_state(state, center, direction):
    normalized_state = rotate_state(center_state(state, center), direction)
    return normalized_state


def convert_action_to_direction(action, cur_direction):
    """
    Gives an updated direction, given action and current direction
    """
    if action == FORWARD_ACTION:
        return cur_direction

    elif action == RIGHT_ACTION:
        if cur_direction == UP:
            return RIGHT
        elif cur_direction == RIGHT:
            return DOWN
        elif cur_direction == LEFT:
            return UP
        elif cur_direction == DOWN:
            return LEFT

    elif action == LEFT_ACTION:
        if cur_direction == UP:
            return LEFT
        elif cur_direction == RIGHT:
            return UP
        elif cur_direction == LEFT:
            return DOWN
        elif cur_direction == DOWN:
            return RIGHT


def get_next_location(loc, direction):
    y, x = loc
    n_y, n_x = y, x
    if direction == UP:
        n_y = (y - 1) % GAME_HEIGHT
    elif direction == DOWN:
        n_y = (y + 1) % GAME_HEIGHT
    elif direction == RIGHT:
        n_x = (x + 1) % GAME_WIDTH
    elif direction == LEFT:
        n_x = (x - 1) % GAME_WIDTH
    # else:
    #     assert 0

    next_loc = n_y, n_x
    return next_loc
