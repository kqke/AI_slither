from keras import Sequential
from keras.layers import Convolution2D, Activation, Dense, Flatten
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np
import time
import os

from players.base_player import BasePlayer
from constants import *
from config import *
from utils import get_greedy_action


DIRECTION_TO_N_ROT90 = {
    UP: 0,
    RIGHT: 1,
    DOWN: 2,
    LEFT: 3
}


class NNPlayer(BasePlayer):
    def __init__(self, pid, head):
        super().__init__(pid, head)
        self.input_shape = (12,)
        self.model = self.build_model()
        self.prev_state = -1
        self.prev_score = -1
        self.prev_q_values = -1
        self.action_index = -1
        self.batch = []
        self.n_batches = 0

        if LOAD_MODEL:
            print("loading model: {}".format(LOAD_MODEL_FILE_NAME))
            self.model = load_model(os.path.join(MODELS_DIR, LOAD_MODEL_FILE_NAME))

    @staticmethod
    def get_type():
        return NN_PLAYER

    def pre_action(self, game):
        self.prev_state = self.extract_model_input(game)
        self.prev_score = self.get_score()

    def get_action(self, game):
        q_values = self.model.predict(self.prev_state)
        self.prev_q_values = q_values

        rand = np.random.random()
        if rand < EPSILON_GREEDY:
            action_index = np.random.randint(N_ACTIONS)
        else:
            action_index = np.argmax(q_values)
        self.action_index = action_index
        action = ACTIONS[action_index]

        print(game.get_state())
        print(q_values)
        print(self.extract_model_input(game))
        print()

        return action

        # # todo tmp
        # greedy_action = get_greedy_action(game, self.head, self.direction)
        # return greedy_action

    def post_action(self, game):
        if not TRAIN_MODEL:
            return

        cur_state = self.extract_model_input(game)
        reward = self.get_score() - self.prev_score
        sample = (self.prev_state, self.action_index, reward, cur_state)
        self.batch.append(sample)

        if len(self.batch) == BATCH_SIZE:
            x = np.zeros((BATCH_SIZE,) + self.input_shape)
            y = np.zeros((BATCH_SIZE, N_ACTIONS))

            for i in range(len(self.batch)):
                state_t, action_index, reward, state_t1 = self.batch[i]
                q_values_t1 = self.model.predict(state_t1)

                x[i] = state_t
                y[i] = self.prev_q_values  # loss is affected only by action_index value
                y[i, action_index] = reward + GAMMA * np.max(q_values_t1)

            loss = self.model.train_on_batch(x, y)
            self.batch = []
            self.n_batches += 1

            if self.n_batches % PRINT_LOSS_BATCH_ITERATIONS == 0:
                print("loss = {:.3f}".format(loss))

            if SAVE_MODEL:
                if self.n_batches % SAVE_MODEL_BATCH_ITERATIONS == 0:
                    # todo tmp
                    print("saving model: {}".format(time.strftime("%Y-%m-%d-%H-%M-%S")))  # todo rm
                    model_fn = "{}.h5".format(time.strftime("%Y-%m-%d-%H-%M-%S"))
                    self.model.save(os.path.join(MODELS_DIR, model_fn))

    # CNN impl.
    def build_model(self):
        model = Sequential()
        model.add(Dense(N_ACTIONS, input_shape=self.input_shape))
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss="mean_squared_error", optimizer=adam)
        # print(model.summary())  # todo
        return model

    def align_state(self, state):
        # todo validate
        n_rot90 = DIRECTION_TO_N_ROT90[self.direction]
        aligned_state = np.rot90(state, n_rot90)
        return aligned_state

    def extract_model_input(self, game):
        aligned_state = self.align_state(game.get_state())
        y, x = np.where(aligned_state == game.get_head_mark(self.pid))
        assert x.shape == y.shape == (1,)
        y = y[0]
        x = x[0]

        model_input = np.zeros(self.input_shape[0])

        model_input[0] = aligned_state[(x-1) % game.get_width(), y] == FREE_SQUARE_MARK
        model_input[1] = aligned_state[x, (y-1) % game.get_height()] == FREE_SQUARE_MARK
        model_input[2] = aligned_state[(x+1) % game.get_width(), y] == FREE_SQUARE_MARK

        model_input[3] = aligned_state[(x-2) % game.get_width(), y] == FREE_SQUARE_MARK
        model_input[4] = aligned_state[x, (y-2) % game.get_height()] == FREE_SQUARE_MARK
        model_input[5] = aligned_state[(x+2) % game.get_width(), y] == FREE_SQUARE_MARK

        model_input[6] = aligned_state[(x-1) % game.get_width(), y] == FOOD_MARK
        model_input[7] = aligned_state[x, (y-1) % game.get_height()] == FOOD_MARK
        model_input[8] = aligned_state[(x+1) % game.get_width(), y] == FOOD_MARK

        model_input[9] = aligned_state[(x-2) % game.get_width(), y] == FOOD_MARK
        model_input[10] = aligned_state[x, (y-2) % game.get_height()] == FOOD_MARK
        model_input[11] = aligned_state[(x+2) % game.get_width(), y] == FOOD_MARK

        model_input = model_input[np.newaxis, :]

        return model_input

