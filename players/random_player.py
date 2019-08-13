import numpy as np

from players.base_player import BasePlayer
from constants import *

FORWARD_ACTION_P = 0.7
OTHER_ACTION_P = (1 - FORWARD_ACTION_P) / 2
ACTION_DISTRIBUTION = [FORWARD_ACTION_P, OTHER_ACTION_P, OTHER_ACTION_P]


class RandomPlayer(BasePlayer):
    @staticmethod
    def get_type():
        return RANDOM_PLAYER

    def get_action(self, player):
        action = np.random.choice(ACTIONS, p=ACTION_DISTRIBUTION)
        return action
