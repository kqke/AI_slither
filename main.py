from constants import *
from config import *
from utils import *
from game import Game


clean_records()

players = {
    CNN_PLAYER: 1,
    # NN_PLAYER: 1,
    GREEDY_PLAYER: 1,
    RANDOM_PLAYER: 1,
    # MANUAL_PLAYER: 1,
}

game = Game(players)
game.run()
