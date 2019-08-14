from keras import Sequential
from keras.layers import Convolution2D, Activation, Dense, Flatten
from keras.optimizers import Adam

import numpy as np


from players.deep_q_player import DeepQPlayer
from constants import *
from config import *

DIRECTION_TO_N_ROT90 = {
    UP: 0,
    RIGHT: 1,
    DOWN: 2,
    LEFT: 3
}


class CNNPlayer(DeepQPlayer):

    def __init__(self, pid, head):
        super().__init__(pid, head)
        self.input_shape = (GAME_HEIGHT, GAME_WIDTH, N_INPUT_CHANNELS)

    @staticmethod
    def get_type():
        return CNN_PLAYER

    # CNN impl.
    def build_model(self):
        model = Sequential()
        model.add(Convolution2D(32, (7, 7), strides=(1, 1), input_shape=self.input_shape))
        model.add(Activation("relu"))
        model.add(Convolution2D(8, (7, 7), strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(32))
        model.add(Activation("relu"))
        model.add(Dense(N_ACTIONS))
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss="mean_squared_error", optimizer=adam)
        print(model.summary())  # todo
        return model

    def extract_model_input(self, game):
        # aligning and centering head
        norm_state = self.normalize_state(game)

        # head isn't modeled since it's centered
        model_input = np.zeros(self.input_shape)
        model_input[:, :, 0] = norm_state == FOOD_MARK  # food
        model_input[:, :, 1] = norm_state == self.pid  # self body
        model_input[:, :, 2] = np.isin(norm_state, self.others_head_marks)  # other heads
        model_input[:, :, 3] = np.isin(norm_state, self.others_body_marks)  # other bodys
        # todo add another map of other heads
        model_input = model_input[np.newaxis, :]
        return model_input
