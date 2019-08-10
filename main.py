from constants import *
from config import *
from game import Game


players = {
    CNN_PLAYER: 1,
    # RANDOM_PLAYER: 1,
    # GREEDY_PLAYER: 1,
}
game = Game(GAME_WIDTH, GAME_HEIGHT, players)
game.run(1000000)
