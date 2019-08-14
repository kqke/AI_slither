from keras import Sequential
from keras.layers import Convolution2D, Activation, Dense, Flatten
from keras.optimizers import Adam


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