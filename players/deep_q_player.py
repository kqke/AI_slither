import numpy as np
import time
import os
from keras.models import load_model

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


class DeepQPlayer(BasePlayer):
    def __init__(self, name, pid, head, input_shape, params):
        super().__init__(name, pid, head)

        self.input_shape = input_shape
        self.params = params

        self.model = self.build_model()

        self.prev_score = -1
        self.prev_model_inputs = []
        self.prev_model_input = -1
        self.batch = []
        self.n_batches = 0

        self.others_head_marks = []
        self.others_body_marks = []

        self.records["loss"] = []
        self.loss = -999

        if self.params["load_model"]:
            print("loading model: {}".format(self.params["load_model_fn"]))
            self.model = load_model(os.path.join(MODELS_DIR, self.params["load_model_fn"]))

    # virtual
    def build_model(self):
        pass

    # virtual
    def extract_model_inputs(self, game):
        pass

    def init(self, game):
        self.others_head_marks = game.get_head_marks()
        self.others_head_marks.remove(game.get_head_mark(self.pid))

        self.others_body_marks = game.get_body_marks()
        self.others_body_marks.remove(self.pid)

    def pre_action(self, game):
        self.prev_score = self.get_score()
        state = game.get_state()
        self.prev_model_inputs = self.extract_model_inputs(state)

    def get_action(self, game):
        rand = np.random.random()
        if rand < self.params["epsilon_greedy"]:
            action_index = np.random.randint(N_ACTIONS)
        else:
            q_values = self.predict_q_values(self.prev_model_inputs)

            action_index = np.argmax(q_values)
            # action_index = get_greedy_action_index(game, self.head, self.direction)

        self.prev_model_input = self.prev_model_inputs[action_index]
        action = ACTIONS[action_index]
        return action

    def predict_q_values(self, model_inputs):
        assert len(model_inputs) == N_ACTIONS

        q_values = []
        for model_input in model_inputs:
            a_q_value = self.model.predict(model_input)
            q_values.append(a_q_value)
        return q_values

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
                    y[i] = reward + self.params["gamma"] * np.max(cur_q_values)

            # todo redesign
            if self.params["train_model"]:
                self.loss = self.model.train_on_batch(x, y)

            self.batch = []
            self.n_batches += 1

            if self.params["save_model"]:
                if self.n_batches % self.params["save_model_batch_iterations"] == 0:
                    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
                    print("saving model: {}".format(time_str))
                    model_fn = "{}_{}.h5".format(self.name, time_str)
                    self.model.save(os.path.join(MODELS_DIR, model_fn))

    def update_records(self):
        super().update_records()
        if self.params["train_model"]:
            self.records["loss"].append(self.loss)
