from keras import Sequential
from keras.layers import Convolution2D, Activation, Dense, Flatten
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np
import os
import time

from players.base_player import BasePlayer
from utils import *
from constants import *
from config import *


class CNNPlayer(BasePlayer):
    def __init__(self, pid, head):
        super().__init__(pid, head)

        self.input_shape = (GAME_HEIGHT, GAME_WIDTH, N_INPUT_CHANNELS)

        self.model = self.build_model()

        self.prev_score = -1
        self.prev_model_inputs = []
        self.prev_model_input = -1
        self.batch = []
        self.n_batches = 0

        self.others_head_marks = set()
        self.others_body_marks = set()

        self.records["loss"] = []
        self.loss = -999

        if LOAD_MODEL:
            print("loading model: {}".format(LOAD_MODEL_FILE_NAME))
            self.model = load_model(os.path.join(MODELS_DIR, LOAD_MODEL_FILE_NAME))

    @staticmethod
    def get_type():
        return CNN_PLAYER

    def init(self, game):
        self.others_head_marks = game.get_head_marks()
        self.others_head_marks.remove(game.get_head_mark(self.pid))

        self.others_body_marks = game.get_body_marks()
        self.others_body_marks.remove(self.pid)

    # CNN impl.
    def build_model(self):
        model = Sequential()
        model.add(Convolution2D(32, (7, 7), strides=(1, 1), input_shape=self.input_shape))
        model.add(Activation("relu"))
        model.add(Convolution2D(8, (7, 7), strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(16))
        model.add(Activation("relu"))
        model.add(Dense(1))
        adam = Adam(lr=LEARNING_RATE)
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

        # todo uc
        model_input[:, :, 0] = (norm_state == FOOD_MARK)  # food
        model_input[:, :, 1] = (norm_state == self.pid)  # self body
        model_input[:, :, 2] = np.isin(norm_state, self.others_head_marks)  # other heads
        model_input[:, :, 3] = np.isin(norm_state, self.others_body_marks)  # other bodys

        model_input = model_input[np.newaxis, :]
        return model_input

    def predict_q_values(self, model_inputs):
        assert len(model_inputs) == N_ACTIONS

        q_values = []
        for model_input in model_inputs:
            a_q_value = self.model.predict(model_input)
            q_values.append(a_q_value)
        return q_values

    def pre_action(self, game):
        self.prev_score = self.get_score()
        state = game.get_state()
        self.prev_model_inputs = self.extract_model_inputs(state)

    def get_action(self, game):
        rand = np.random.random()
        if rand < EPSILON_GREEDY:
            action_index = np.random.randint(N_ACTIONS)
        else:
            q_values = self.predict_q_values(self.prev_model_inputs)
            # print("q: {}".format(q_values))  # todo rm

            action_index = np.argmax(q_values)
            # action_index = get_greedy_action_index(game, self.head, self.direction)

        self.prev_model_input = self.prev_model_inputs[action_index]
        action = ACTIONS[action_index]
        return action

    def post_action(self, game):
        state = game.get_state()
        cur_model_inputs = self.extract_model_inputs(state)
        reward = self.get_score() - self.prev_score
        sample = (self.prev_model_input, reward, cur_model_inputs)
        self.batch.append(sample)

        if len(self.batch) == BATCH_SIZE:
            x = np.zeros((BATCH_SIZE,) + self.input_shape)
            y = np.zeros(BATCH_SIZE)

            for i in range(len(self.batch)):
                prev_model_input, reward, cur_model_inputs = self.batch[i]
                cur_q_values = self.predict_q_values(cur_model_inputs)

                x[i] = prev_model_input

                # todo review
                if reward < 0:  # snake is dead
                    y[i] = reward
                else:
                    y[i] = reward + GAMMA * np.max(cur_q_values)

            # todo redesign
            if TRAIN_MODEL:
                self.loss = self.model.train_on_batch(x, y)

            self.batch = []
            self.n_batches += 1

            if SAVE_MODEL:
                if self.n_batches % SAVE_MODEL_BATCH_ITERATIONS == 0:
                    # todo tmp
                    print("saving model: {}".format(time.strftime("%Y-%m-%d-%H-%M-%S")))  # todo rm
                    model_fn = "{}.h5".format(time.strftime("%Y-%m-%d-%H-%M-%S"))
                    self.model.save(os.path.join(MODELS_DIR, model_fn))

    def update_records(self):
        super().update_records()
        if TRAIN_MODEL:
            self.records["loss"].append(self.loss)

