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
from utils import *


DIRECTION_TO_N_ROT90 = {
    UP: 0,
    RIGHT: 1,
    DOWN: 2,
    LEFT: 3
}

# FEATURES:
# - closest food/food cluster / distances to food
# - closest collision / number of collisions 1 step away
# - whether food is eaten


class NNPlayer(DeepQPlayer):

    def __init__(self, pid, head):
        super().__init__(pid, head)
        self.input_shape = NN_INPUT_SHAPE

    @staticmethod
    def get_type():
        return NN_PLAYER

    # NN impl.
    # todo
    # not ready yet
    def build_model(self):
        model = Sequential()
        model.add(Dense(N_ACTIONS, input_shape=self.input_shape))
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss="mean_squared_error", optimizer=adam)
        # print(model.summary())  # todo
        return model

    def extract_model_input(self, game):
        food_dist = self.get_closest_food(game)
        coll_dist = self.get_closest_collision(game)
        eat = self.is_food_eaten()
        return np.array([food_dist, coll_dist, eat])

    def get_closest_food(self, game):
        foods = game.get_food()
        min_dist = (np.inf, np.inf)
        for food in foods:
            dist = l1_distance(self.head, food)
            if dist < min_dist:
                min_dist = food
        return min_dist

    def get_closest_collision(self, game):
        players = game.get_players()
        min_dist = (np.inf, np.inf)
        for player in players:
            locations = player.get_locations()
            for location in locations:
                dist = l1_distance(self.head, location)
                if dist < min_dist:
                    min_dist = location
        return min_dist

    def is_food_eaten(self, game):
        # not sure about impl.
        return True
