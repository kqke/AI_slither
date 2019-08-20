from keras import Sequential
from keras.layers import Convolution2D, Activation, Dense, Flatten
from keras.optimizers import Adam

from players.deep_q_player import DeepQPlayer
from utils import *
from constants import *
from config import *


class CNNPlayer(DeepQPlayer):
    def __init__(self, name, pid, head):
        input_shape = (GAME_HEIGHT, GAME_WIDTH, CNN_PARAMS["n_input_channels"])
        super().__init__(name, pid, head, input_shape, CNN_PARAMS)

    @staticmethod
    def get_type():
        return CNN_PLAYER

    # CNN impl.
    def build_model(self):
        # 15x15 game, 15 food
        # model = Sequential()
        # model.add(Convolution2D(32, (7, 7), strides=(1, 1), input_shape=self.input_shape))
        # model.add(Activation("relu"))
        # model.add(Convolution2D(8, (7, 7), strides=(1, 1)))
        # model.add(Activation("relu"))
        # model.add(Flatten())
        # model.add(Dense(16))
        # model.add(Activation("relu"))
        # model.add(Dense(1))
        # adam = Adam(lr=LEARNING_RATE)
        # model.compile(loss="mean_squared_error", optimizer=adam)
        # model.summary()
        # return model

        # 21x21 game, 15 food
        model = Sequential()
        model.add(Convolution2D(64, (9, 9), strides=(1, 1), input_shape=self.input_shape))
        model.add(Activation("relu"))
        model.add(Convolution2D(8, (9, 9), strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(16))
        model.add(Activation("relu"))
        model.add(Dense(1))
        adam = Adam(lr=CNN_PARAMS["learning_rate"])
        model.compile(loss="mean_squared_error", optimizer=adam)
        model.summary()
        return model

    def extract_model_inputs(self, state):
        centered_state = center_state(state, self.head)

        model_inputs = []
        for action in ACTIONS:
            a_direction = convert_action_to_direction(action, self.direction)
            a_state = rotate_state(centered_state, a_direction)
            a_model_input = self.extract_model_input(a_state)
            model_inputs.append(a_model_input)
        return model_inputs

    def extract_model_input(self, norm_state):
        # head isn't modeled since it's centered
        model_input = np.zeros(self.input_shape)

        model_input[:, :, 0] = (norm_state == FOOD_MARK)  # food
        model_input[:, :, 1] = (norm_state == self.pid)  # self body
        model_input[:, :, 2] = np.isin(norm_state, self.others_head_marks)  # other heads
        model_input[:, :, 3] = np.isin(norm_state, self.others_body_marks)  # other bodies

        model_input = model_input[np.newaxis, :]
        return model_input
