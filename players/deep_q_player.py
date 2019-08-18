import numpy as np

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

        self.input_shape = input_shape

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
        if rand < EPSILON_GREEDY:
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
                    y[i] = reward + GAMMA * np.max(cur_q_values)

            # todo redesign
            if TRAIN_MODEL:
                self.loss = self.model.train_on_batch(x, y)

            self.batch = []
            self.n_batches += 1

    def update_records(self):
        super().update_records()
        if TRAIN_MODEL:
            self.records["loss"].append(self.loss)
