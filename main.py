from utils import *
from game import Game

clean_records()

players = {
    CNN_PLAYER: "CNN",
    # NN_PLAYER: "NN",
    GREEDY_PLAYER: "GREEDY",
    RANDOM_PLAYER: "RANDOM",
    # MANUAL_PLAYER: "MANUAL",
}

game = Game(players)
game.run()
