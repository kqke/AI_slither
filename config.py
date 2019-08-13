# GAME
GUI = 0
N_ITERATIONS = 10000000
STARTING_LENGTH = 3

GAME_WIDTH = 15
GAME_HEIGHT = 15
N_FOOD = 10
FOOD_SIZE_INC = 1

# scores
SCORE_FOOD = 1  # todo
SCORE_DEAD = -3  # todo
SCORE_KILLING = 0  # todo


# CNN
TRAIN = 1

EPSILON_GREEDY = 0.1 if TRAIN else 0  # todo
GAMMA = .9

LEARNING_RATE = .01  # todo
BATCH_SIZE = 64
N_INPUT_CHANNELS = 4

TRAIN_MODEL = 1  #1 if TRAIN else 0  # todo

SAVE_MODEL = 1
PRINT_LOSS_BATCH_ITERATIONS = 100
SCORE_SUMMARY_BATCH_ITERATION = 300 if TRAIN else 20
SAVE_MODEL_BATCH_ITERATIONS = 3000

LOAD_MODEL = 0
LOAD_MODEL_FILE_NAME = "2019-08-13-04-04-28.h5"



# todo
# look at EPSILON_GREEDY param
# look at cnn_player.py get_action function! (on/off policy)
# enhance CNN
# try without conv, only dense