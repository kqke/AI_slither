from players.base_player import BasePlayer
from constants import *
from utils import get_greedy_action


class GreedyPlayer(BasePlayer):
    @staticmethod
    def get_type():
        return GREEDY_PLAYER

    def get_action(self, game):
        greedy_action = get_greedy_action(game, self.head, self.direction)
        return greedy_action
