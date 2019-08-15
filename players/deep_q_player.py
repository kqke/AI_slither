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
from utils import get_greedy_action_index

DIRECTION_TO_N_ROT90 = {
    UP: 0,
    RIGHT: 1,
    DOWN: 2,
    LEFT: 3
}


class DeepQPlayer(BasePlayer):
    def __init__(self, pid, head, input_shape):
        super().__init__(pid, head)

        # todo
        # maybe not so good...
        # this attribute is changed in inheriting classes,
        # but is used in post_action func
        self.input_shape = input_shape

        self.center_y = GAME_HEIGHT // 2
        self.center_x = GAME_WIDTH // 2
        self.model = self.build_model()
        self.prev_state = -1
        self.prev_score = -1
        self.prev_q_values = -1
        self.action_index = -1
        self.batch = []
        self.n_batches = 0
        self.others_head_marks = set()
        self.others_body_marks = set()

        self.records["loss"] = []
        self.loss = -999

        self.tmp = -1  # todo rm

        if LOAD_MODEL:
            print("loading model: {}".format(LOAD_MODEL_FILE_NAME))
            self.model = load_model(os.path.join(MODELS_DIR, LOAD_MODEL_FILE_NAME))

    # virtual
    def build_model(self):
        pass

    # virtual
    def extract_model_input(self, game):
        pass

    def init(self, game):
        self.others_head_marks = game.get_head_marks()
        self.others_head_marks.remove(game.get_head_mark(self.pid))

        self.others_body_marks = game.get_body_marks()
        self.others_body_marks.remove(self.pid)

    def pre_action(self, game):
        self.tmp = game.get_state()  # todo rm
        self.prev_state = self.extract_model_input(game)
        self.prev_score = self.get_score()

    def get_action(self, game):
        q_values = self.model.predict(self.prev_state)
        self.prev_q_values = q_values

        # print("q: {}".format(q_values))

        # # todo uc
        rand = np.random.random()
        if rand < EPSILON_GREEDY:
            action_index = np.random.randint(N_ACTIONS)
        else:
            action_index = np.argmax(q_values)
            # action_index = get_greedy_action_index(game, self.head, self.direction)
        self.action_index = action_index
        action = ACTIONS[action_index]
        return action

    def post_action(self, game):
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

                # todo tmp
                # y[i, action_index] = reward + GAMMA * np.max(q_values_t1)

                # todo review
                if reward < 0:  # snake is dead
                    y[i, action_index] = reward
                else:
                    y[i, action_index] = reward + GAMMA * np.max(q_values_t1)

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

    def normalize_state(self, game):
        # align state
        n_rot90 = DIRECTION_TO_N_ROT90[self.direction]
        aligned_state = np.rot90(game.get_state(), n_rot90)

        # roll s.t. head is in center
        y, x = np.where(aligned_state == game.get_head_mark(self.pid))
        # todo rm
        if not (x.shape == y.shape == (1,)):
            print(self.pid)
            print(game.get_head_mark(self.pid))
            print(x)
            print(y)
            print(aligned_state)
        assert x.shape == y.shape == (1,)
        head_y = y[0]
        head_x = x[0]
        norm_state = np.roll(np.roll(aligned_state, self.center_y - head_y, axis=0), self.center_x - head_x, axis=1)
        return norm_state
