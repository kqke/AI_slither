from players.base_player import BasePlayer
from constants import *


VALID_INPUTS = {
    "j": LEFT_ACTION,
    "i": FORWARD_ACTION,
    "l": RIGHT_ACTION
}


class ManualPlayer(BasePlayer):
    @staticmethod
    def get_type():
        return MANUAL_PLAYER

    def get_action(self, player):
        manual_input = ""
        while manual_input not in VALID_INPUTS:
            manual_input = input("j - left, i - forward, j - right")
        action = VALID_INPUTS[manual_input]
        return action
