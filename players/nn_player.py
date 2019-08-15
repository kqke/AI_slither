from keras import Sequential
from keras.layers import Convolution2D, Activation, Dense, Flatten
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.cluster import MeanShift
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
        food_cluster_dist = self.get_cluster_dist(game)
        coll_dist = self.get_closest_collision(game)
        # eat = self.is_food_eaten(game)
        return np.array([food_dist, food_cluster_dist, coll_dist]).reshape(NN_INPUT_SHAPE)

    def get_closest_food(self, game):
        foods = game.get_food()
        return min(l1_distance(self.head, food) for food in foods)

    def get_cluster_dist(self, game):
        food = np.asarray(list(game.get_food()))
        clusters = MeanShift().fit(food)
        centers = clusters.cluster_centers_
        return min(l1_distance(self.head, center) for center in centers)

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

    # def is_food_eaten(self, game):
    #     # not sure about impl.
    #     return True
