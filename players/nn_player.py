from keras import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam

from players.deep_q_player import DeepQPlayer
from utils import *


DIRECTION_TO_N_ROT90 = {
    UP: 0,
    RIGHT: 1,
    DOWN: 2,
    LEFT: 3
}

class NNPlayer(DeepQPlayer):

    def __init__(self, pid, head):
        super().__init__(pid, head, NN_INPUT_SHAPE)
        self.others_marks = self.others_body_marks | self.others_head_marks

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
        # print(model.summary())  # todo
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
        food_sum = np.sum(rot_state[:GAME_CENTER_Y, (GAME_CENTER_X - RADIUS):(GAME_CENTER_Y + RADIUS)] == FOOD_MARK)
        block_sum = np.sum(np.isin(rot_state[:GAME_CENTER_Y, (GAME_CENTER_X - RADIUS):(GAME_CENTER_X + RADIUS)],
                                   self.others_marks))
        model_input = np.array([food, block, food_sum, block_sum]).reshape(NN_INPUT_SHAPE)
        model_input = model_input[np.newaxis, :]
        return model_input
