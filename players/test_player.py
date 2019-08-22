import numpy as np

from players.base_player import BasePlayer
from constants import *
from config import *


class TestPlayer(BasePlayer):

    def __init__(self, pid, head, params):
        super().__init__("TEST", pid, head, leftover=params['leftover'])
        self.head = params['head']
        self.direction = params['direction']
        self.actions = params['actions']
        self.action_counter = 0
        self.turn_counter = 0
        self.cur_action = self.actions[self.action_counter]

    @staticmethod
    def get_type():
        return TEST_PLAYER

    def get_action(self, game):
        action = FORWARD_ACTION
        if self.turn_counter > self.cur_action[COUNTER]:
            self.action_counter = (self.action_counter + 1) % len(self.actions)
            self.cur_action = self.actions[self.action_counter]
            self.turn_counter = 0
            action = self.cur_action[ACTION]
        self.turn_counter += 1
        return action


