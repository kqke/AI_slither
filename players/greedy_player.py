import numpy as np

from players.base_player import BasePlayer
from constants import *
from utils.geo import l1_distance


class GreedyPlayer(BasePlayer):
    @staticmethod
    def get_type():
        return GREEDY_PLAYER

    def get_action(self, game):
        food = game.get_food()
        if len(food) == 0:
            return FORWARD_ACTION

        # find the nearest food
        food_and_distances = []
        for food_xy in food:
            dist = l1_distance(food_xy, self.head)
            food_and_distances.append((food_xy, dist))
        min_food = min(food_and_distances, key=lambda fd: fd[1])[0]

        # find the action which get us closest the the nearest food
        state = game.get_state()
        actions_and_distances = []
        for action in ACTIONS:
            action_direction = game.convert_action_to_direction(action, self.direction)
            action_loc = game.get_next_location(self.head, action_direction)

            # avoid obvious collisions
            if state[action_loc] in [FOOD, FREE_SQUARE]:
                dist = l1_distance(min_food, action_loc)
            else:  # snake's body or another snake
                dist = float("inf")

            actions_and_distances.append((action, dist))

        greedy_action = min(actions_and_distances, key=lambda ad: ad[1])[0]
        return greedy_action
