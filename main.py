from utils import *
from game import Game

clean_records()

players = {
    CNN_PLAYER: "CNN",
    NN_PLAYER: "NN",
    GREEDY_PLAYER: "GREEDY",
    # RANDOM_PLAYER: "RANDOM",
    # MANUAL_PLAYER: "MANUAL",
    # TEST_1: S_RECT_PARAMS,
    # TEST_2: A_PARAMS,
    # TEST_3: B_RECT_PARAMS
}
if __name__ == '__main__':
    players = {
        CNN_PLAYER: "CNN",
        NN_PLAYER: "NN",
        GREEDY_PLAYER: "GREEDY",
        # RANDOM_PLAYER: "RANDOM",
        # MANUAL_PLAYER: "MANUAL",
    }
    game = Game(players)
    game.run()
