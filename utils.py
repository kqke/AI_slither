import numpy as np

from constants import *


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


def get_greedy_action_index(game, head, direction):
    food = game.get_food()
    if len(food) == 0:
        return ACTIONS.index(FORWARD_ACTION)

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
        action_direction = game.convert_action_to_direction(action, direction)
        action_loc = game.get_next_location(head, action_direction, game.get_height(), game.get_width())

        # avoid obvious collisions
        if state[action_loc] in [FOOD_MARK, FREE_SQUARE_MARK]:
            dist = l1_distance(min_food, action_loc)
        else:  # snake's body or another snake
            dist = float("inf")

        actions_indices_and_distances.append((i, dist))

    greedy_action_index = min(actions_indices_and_distances, key=lambda ad: ad[1])[0]
    return greedy_action_index
