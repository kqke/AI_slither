from players.base_player import BasePlayer
from constants import *


class GreedyPlayer(BasePlayer):
    @staticmethod
    def get_type():
        return GREEDY_PLAYER

    def get_action(self, game):
        pass
