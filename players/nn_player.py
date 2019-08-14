from keras import Sequential
from keras.layers import Convolution2D, Activation, Dense, Flatten
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np
import time
import os

from players.deep_q_player import DeepQPlayer
from constants import *
from config import *
from utils import get_greedy_action


DIRECTION_TO_N_ROT90 = {
    UP: 0,
    RIGHT: 1,
    DOWN: 2,
    LEFT: 3
}


class NNPlayer(DeepQPlayer):

    @staticmethod
    def get_type():
        return NN_PLAYER

    # NN impl.
    def build_model(self):
        model = Sequential()
        model.add(Dense(N_ACTIONS, input_shape=self.input_shape))
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss="mean_squared_error", optimizer=adam)
        # print(model.summary())  # todo
        return model
