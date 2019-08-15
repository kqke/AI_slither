# GAME
GUI = 0
N_ITERATIONS = 10000000
STARTING_LENGTH = 3

GAME_WIDTH = 15
GAME_HEIGHT = 15
N_FOOD = 15
FOOD_SIZE_INC = 1

# scores
SCORE_FOOD = 1  # todo
SCORE_DEAD = -3  # todo
SCORE_KILLING = 3  # todo

# CNN
EPSILON_GREEDY = 0  #todo
GAMMA = .95  # todo was .9

LEARNING_RATE = .0001  # todo
BATCH_SIZE = 64
N_INPUT_CHANNELS = 4

TRAIN_MODEL = 1

SAVE_MODEL = 1
SAVE_MODEL_BATCH_ITERATIONS = 1000

LOAD_MODEL = 1
LOAD_MODEL_FILE_NAME = "2019-08-15-05-09-47.h5"

# records
PRINT_RECORDS = True
PRINT_RECORDS_BATCH_ITERATIONS = 200
SAVE_RECORDS = False
SAVE_RECORDS_BATCH_ITERATIONS = 300

# NN

N_FEATURES = 3
NN_INPUT_SHAPE = (1, N_FEATURES)



# todo
# look at EPSILON_GREEDY param
# look at cnn_player.py get_action function! (on/off policy)
# enhance CNN
# try without conv, only dense
# if dead, pehaps change q target to reward w.o. q of next state?
# go over todos
# rm prints or organize them
# think about framework
# game sometimes crashes when starts
# no need for greedy epsilon since game is already quite stochastic
