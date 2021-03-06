# GAME
GUI = 1
GUI_DELAY = 15
N_ITERATIONS = 10000000
STARTING_LENGTH = 3

GAME_WIDTH = 21
GAME_HEIGHT = 21
GAME_SHAPE = (GAME_HEIGHT, GAME_WIDTH)
assert GAME_WIDTH % 2 == 1 and GAME_HEIGHT % 2 == 1  # should be odd to center head
GAME_CENTER_Y = GAME_HEIGHT // 2
GAME_CENTER_X = GAME_WIDTH // 2

N_FOOD = 15
FOOD_SIZE_INC = 1

# RL params
SCORE_FOOD = 1
SCORE_DEAD = -5
SCORE_KILLING = 3

GAMMA = .95

# CNN & NN models
BATCH_SIZE = 64

CNN_PARAMS = {
    "epsilon_greedy": 0,
    "learning_rate": .0001,
    "n_input_channels": 4,
    "train_model": 0,
    "save_model": 0,
    "save_model_batch_iterations": 20,
    "load_model": 1,
    "load_model_fn": "CNN_2019-08-19-16-24-29.h5",
}

NN_PARAMS = {
    "epsilon_greedy": 0.001,
    "learning_rate": .0001,
    "n_features": 4,
    "radius": 3,
    "train_model": 0,
    "save_model": 0,
    "save_model_batch_iterations": 1000,
    "load_model": 1,
    "load_model_fn": "NN_2019-08-19-16-17-14.h5",
}

from constants import *

A_PARAMS = {
    "head": (GAME_HEIGHT - 3, 2),
    "direction": UP,
    "leftover": 40,
    "actions": [(10, RIGHT_ACTION),
                (7, RIGHT_ACTION),
                (10, RIGHT_ACTION),
                (2, RIGHT_ACTION),
                (3, RIGHT_ACTION),
                (1, LEFT_ACTION),
                (3, LEFT_ACTION),
                (2, RIGHT_ACTION),
                ]
}

B_RECT_PARAMS = {
    "head": (GAME_HEIGHT - 3, GAME_WIDTH - 8),
    "direction": UP,
    "leftover": 30,
    "actions": [(10, RIGHT_ACTION),
                (5, RIGHT_ACTION),
                ]
}

S_RECT_PARAMS = {
    "head": (GAME_HEIGHT - 9, 6),
    "direction": UP,
    "leftover": 3,
    "actions": [(0, RIGHT_ACTION),
                ]
}

# records
PRINT_RECORDS = True
PRINT_RECORDS_BATCH_ITERATIONS = 20
SAVE_RECORDS = True
SAVE_RECORDS_BATCH_ITERATIONS = 1000
