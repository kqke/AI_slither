from players.base_player import BasePlayer
from constants import *
from config import *
if GUI:
    import pygame


VALID_INPUTS = {
    "j": LEFT_ACTION,
    "i": FORWARD_ACTION,
    "l": RIGHT_ACTION
}

ARROW_DICT = {
    UP: {
        UP: FORWARD_ACTION,
        LEFT: LEFT_ACTION,
        DOWN: FORWARD_ACTION,
        RIGHT: RIGHT_ACTION
    },
    LEFT: {
        UP: RIGHT_ACTION,
        LEFT: FORWARD_ACTION,
        DOWN: LEFT_ACTION,
        RIGHT: FORWARD_ACTION
    },
    DOWN: {
        UP: FORWARD_ACTION,
        LEFT: RIGHT_ACTION,
        DOWN: FORWARD_ACTION,
        RIGHT: LEFT_ACTION
    },
    RIGHT: {
        UP: LEFT_ACTION,
        LEFT: FORWARD_ACTION,
        DOWN: RIGHT_ACTION,
        RIGHT: FORWARD_ACTION
    }
}


class ManualPlayer(BasePlayer):
    @staticmethod
    def get_type():
        return MANUAL_PLAYER

    def get_action(self, game):
        if GUI:
            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_w] or pressed[pygame.K_UP]:
                return ARROW_DICT[self.direction][UP]
            if pressed[pygame.K_a] or pressed[pygame.K_LEFT]:
                return ARROW_DICT[self.direction][LEFT]
            if pressed[pygame.K_s] or pressed[pygame.K_DOWN]:
                return ARROW_DICT[self.direction][DOWN]
            if pressed[pygame.K_d] or pressed[pygame.K_RIGHT]:
                return ARROW_DICT[self.direction][RIGHT]
            return FORWARD_ACTION

        else:
            manual_input = ""
            while manual_input not in VALID_INPUTS:
                manual_input = input("j - left, i - forward, j - right")
            action = VALID_INPUTS[manual_input]
            return action
