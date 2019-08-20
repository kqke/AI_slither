from utils import *
from game import Game

if __name__ == '__main__':
    clean_records()

    players = {
        CNN_PLAYER: "EVO",
        # NN_PLAYER: "NN",
        GREEDY_PLAYER: "GREEDY",
        RANDOM_PLAYER: "RANDOM",
        # MANUAL_PLAYER: "MANUAL",
    }

    game = Game(players)
    game.run()