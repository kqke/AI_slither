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
    def __init__(self, name, pid, head):
        input_shape = (NN_PARAMS["n_features"],)
        super().__init__(name, pid, head, input_shape, NN_PARAMS)

    @staticmethod
    def get_type():
        return NN_PLAYER

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(Dense(1))
        adam = Adam(lr=self.params["learning_rate"])
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

    def extract_model_input(self, reg_state, norm_state, direction):

        r = self.params["radius"]
        window = norm_state[GAME_CENTER_Y-2*r-1:GAME_CENTER_Y, GAME_CENTER_X-r:GAME_CENTER_X+r+1]
        n_loc = get_next_location(self.head, direction)
        n_square = reg_state[n_loc]

        model_input = np.zeros(self.input_shape)
        model_input[0] = n_square == FOOD_MARK
        model_input[1] = (n_square != FOOD_MARK) & (n_square != FREE_SQUARE_MARK)
        model_input[2] = np.sum(window == FOOD_MARK)
        model_input[3] = np.sum(np.isin(window, self.others_body_marks + self.others_head_marks + [self.pid]))
        # block_head_sum = np.sum(np.isin(window, self.others_head_marks))
        # my_sum = np.sum(window == self.pid)
        # model_input = np.array([food, block, food_sum, block_body_sum, block_head_sum, my_sum]).reshape(self.input_shape)
        model_input = model_input[np.newaxis, :]
        return model_input
