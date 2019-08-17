# GAME
GUI = 1
N_ITERATIONS = 10000000
STARTING_LENGTH = 3

GAME_WIDTH = 21
GAME_HEIGHT = 21
GAME_SHAPE = (GAME_HEIGHT, GAME_WIDTH)
assert GAME_WIDTH % 2 == 1 and GAME_HEIGHT % 2 == 1  # should be odd to center head
GAME_CENTER_Y = GAME_HEIGHT // 2
GAME_CENTER_X = GAME_WIDTH // 2

N_FOOD = 1
FOOD_SIZE_INC = 35

# scores
SCORE_FOOD = 1
SCORE_DEAD = -5
SCORE_KILLING = 3

# CNN
EPSILON_GREEDY = 0  # todo
GAMMA = .95  # todo was .9

LEARNING_RATE = .0001  # todo
BATCH_SIZE = 64
N_INPUT_CHANNELS = 4

TRAIN_MODEL = 1

SAVE_MODEL = 1
SAVE_MODEL_BATCH_ITERATIONS = 1000

LOAD_MODEL = 1
LOAD_MODEL_FILE_NAME = "2019-08-17-16-55-35.h5"

# records
PRINT_RECORDS = True
PRINT_RECORDS_BATCH_ITERATIONS = 200
SAVE_RECORDS = False
SAVE_RECORDS_BATCH_ITERATIONS = 300

# NN

N_FEATURES = 4
NN_INPUT_SHAPE = (N_FEATURES, )

RADIUS = 3


# todo
# look at EPSILON_GREEDY param
# look at cnn_player.py get_action function! (on/off policy)
# enhance CNN
# try without conv, only dense
# if dead, perhaps change q target to reward w.o. q of next state?
# go over todos
# rm prints or organize them
# think about framework
# game sometimes crashes when starts
# no need for greedy epsilon since game is already quite stochastic
# profile code?
# add plots of records
# plot state when an event occurred (long snake, kills, dies, eats a lot food in short time...)
