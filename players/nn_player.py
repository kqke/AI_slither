from keras.models import load_model
from keras import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam

from players.deep_q_player import DeepQPlayer
from utils import *

import time


DIRECTION_TO_N_ROT90 = {
    UP: 0,
    RIGHT: 1,
    DOWN: 2,
    LEFT: 3
}


class NNPlayer(DeepQPlayer):

    def __init__(self, pid, head):
        super().__init__(pid, head, NN_INPUT_SHAPE)
        if LOAD_MODEL:
            print("loading model: {}".format(LOAD_MODEL_FILE_NAME))
            self.model = load_model(os.path.join(NN_MODELS_DIR, LOAD_MODEL_FILE_NAME))

    @staticmethod
    def get_type():
        return NN_PLAYER

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(Dense(1))
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss="mean_squared_error", optimizer=adam)
        model.summary()  # todo
        return model

    def extract_model_inputs(self, state):
        centered_state = center_state(state, self.head)
        model_inputs = []
        for action in ACTIONS:
            a_direction = convert_action_to_direction(action, self.direction)
            a_state = rotate_state(centered_state, a_direction)
            a_model_input = self.extract_model_input(state, a_state, a_direction)
            model_inputs.append(a_model_input)
        return model_inputs

    def extract_model_input(self, reg_state, rot_state, direction):
        n_loc = get_next_location(self.head, direction)
        food, block = 0, 0
        if reg_state[n_loc] == FOOD_MARK:
            food = 1
        elif reg_state[n_loc] != FREE_SQUARE_MARK:
            block = 1
        food_sum = np.sum(rot_state[:GAME_CENTER_Y] == FOOD_MARK)
        block_body_sum = np.sum(np.isin(rot_state[:GAME_CENTER_Y], self.others_body_marks))
        block_head_sum = np.sum(np.isin(rot_state[:GAME_CENTER_Y], self.others_head_marks))
        my_sum = np.sum(rot_state[:GAME_CENTER_Y, (GAME_CENTER_X - RADIUS):(GAME_CENTER_X + RADIUS)] == self.pid)
        model_input = np.array([food, block, food_sum, block_body_sum, block_head_sum, my_sum]).reshape(NN_INPUT_SHAPE)
        model_input = model_input[np.newaxis, :]
        return model_input

    def post_action(self, game):
        super().post_action(game)
        if SAVE_MODEL:
            if self.n_batches % SAVE_MODEL_BATCH_ITERATIONS == 0:
                # todo tmp
                print("saving model: {}".format(time.strftime("%Y-%m-%d-%H-%M-%S")))  # todo rm
                model_fn = "{}.h5".format(time.strftime("%Y-%m-%d-%H-%M-%S"))
                self.model.save(os.path.join(NN_MODELS_DIR, model_fn))
