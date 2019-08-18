from players.base_player import BasePlayer
from constants import *
from utils import get_greedy_action_index


class GreedyPlayer(BasePlayer):
    @staticmethod
    def get_type():
        return GREEDY_PLAYER

    def get_action(self, game):
        greedy_action_index = get_greedy_action_index(game, self.head, self.direction)
        greedy_action = ACTIONS[greedy_action_index]
        return greedy_action
