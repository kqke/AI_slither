# TODO
# replace game_player by base_player - need pid in cnn_player (more intuitive implementation by using inheritance)
# kepp a direction member
# remove p.get_id() - 1 from code - prone to bugs (replace by dict or function if still necessary)
# food should be represented in state
# x corresponds to width and y to height (stick with this convention - o.w. prone to bugs)
# search for todo s ...
# if reaching end state - don't update cnn player
# check if CNN can handle width != height, maybe enforce width = height

import os
import numpy as np
from pickle import dump

from players.cnn_player import CNNPlayer
from players.nn_player import NNPlayer
from players.greedy_player import GreedyPlayer
from players.random_player import RandomPlayer
from players.manual_player import ManualPlayer
from constants import *
from config import *
from utils import sample_bool_matrix
from pygame_snake import play_gui

SCORE_MULTIPLIER = 2
DIRECTION = 6
PLAYER = 5
HEAD_POS = 4
LOCATION = 3
TAIL_POS = 2


class Game:
    """
    Game class
    """

    def __init__(self, width, height, players):
        """
        Initialize game.
        :param width: Width of the board.
        :param height: Height of the board.
        :param players: A dict that contains key-value pairs thar correspond to player type and their amount.
        """
        self.clean_records()

        self._h, self._w = height, width
        self._state = np.full((height, width), FREE_SQUARE_MARK)
        self._check = self._state.copy()
        self._players_dict = dict()
        self._food = set()
        self._turn_number = 1
        self._dead = set()
        self.construct_players(players)
        self.update_board()
        self.update_food()
        self.head_marks = [self.get_head_mark(pid) for pid in self.get_ids()]
        self.body_marks = list(self.get_ids())
        self.init_players()

    def construct_players(self, players):
        """
        Factory method that initializes player instances.
        :param players: A dict that contains key-value pairs thar correspond to player type and their amount.
        """
        pid = 1
        init_state = self.get_state()
        for player in players:
            n = players[player]
            for k in range(n):
                head = sample_bool_matrix(init_state == FREE_SQUARE_MARK)
                init_state[head] = FREE_SQUARE_MARK - 1  # not equal to FREE_SQUARE_MARK
                if player == CNN_PLAYER:
                    self._players_dict[pid] = CNNPlayer(pid, head)
                elif player == NN_PLAYER:
                    self._players_dict[pid] = NNPlayer(pid, head)
                elif player == GREEDY_PLAYER:
                    self._players_dict[pid] = GreedyPlayer(pid, head)
                elif player == RANDOM_PLAYER:
                    self._players_dict[pid] = RandomPlayer(pid, head)
                elif player == MANUAL_PLAYER:
                    self._players_dict[pid] = ManualPlayer(pid, head)
                else:
                    assert 0
                pid += 1

    def init_players(self):
        for player in self.get_players():
            player.init(self)

    def run(self, turns):
        """
        Runs the game for max_turns (specified in constructor) turns.
        """
        if GUI:
            play_gui(self, turns)
        else:
            i = 0
            while i < turns:
                # print(self)
                i += 1
                self.play_turn()

    def play_turn(self):
        """
        Advances the game by one turn.
        """
        self.pre_turn()
        self.move_players()
        self.check_collisions()
        # self.check_enclosure()
        self.update_board()
        self.update_food()
        self.post_turn()
        self.update_players_records()
        self._turn_number += 1

    def pre_turn(self):
        for player in self.get_players():
            player.pre_action(self)

    def move_players(self):
        """
        Move each player to its next position.
        """
        for player in self.get_players():
            self.do_action(player)
            self.check_food(player)

    def do_action(self, player):
        """
        Advances a player according to its action.
        :param player: The player.
        """
        action = player.get_action(self)

        direction = self.convert_action_to_direction(action, player.get_direction())
        player.set_direction(direction)
        n_y, n_x = self.get_next_location(player.get_head(), direction, self._h, self._w)
        new_loc = (n_y, n_x)
        # print("{}: {} -> {} ({})".format(player.get_id(), player.get_head(), new_loc, direction))  # todo verbose
        player.move(new_loc)

    def check_food(self, player):
        """
        Checks whether a snake has eaten a food token in the previous turn,
        if so, its score is updated, and true is returned.
        :return:
        """
        if player.get_head() in self._food:
            player.eat()
            self._food.remove(player.get_head())

    def check_collisions(self):
        """
        Checks whether two snakes have collided, if so, the colliding snake is pronounced dead.
        In head-on collision, both snakes die.
        """
        players = list(self.get_players())
        n_players = len(players)
        for i in range(n_players):
            p1 = players[i]

            # self body to head collision
            if p1.get_head() in p1.get_location_set():
                self._dead.add(p1)

            for j in range(i+1, n_players):
                p2 = players[j]

                # head to body collision
                if p1.get_head() in p2.get_location_set():
                    self._dead.add(p1)
                    p2.kill()

                # head to body collision
                if p2.get_head() in p1.get_location_set():
                    self._dead.add(p2)
                    p1.kill()

                # head to head collision
                if p1.get_head() == p2.get_head():
                    self._dead.add(p1)
                    self._dead.add(p2)
                    p1.kill()
                    p2.kill()

    def update_board(self):
        """
        Generates a numpy array corresponding to the current game state.
        """
        self._state = np.full((self._h, self._w), FREE_SQUARE_MARK)
        # add player marks
        for pid, player in self._players_dict.items():
            if player not in self._dead:
                for loc in player.get_locations():
                    self._state[loc] = pid
                self._state[player.get_head()] = self.get_head_mark(pid)

        # add food marks
        for food in self._food:
            self._state[food] = FOOD_MARK

        # add new spawn snakes
        self.update_dead()

    def update_dead(self):
        for player in self._dead:
            # print("{} is dead!".format(dead.get_id()))  # todo verbose - print state to learn how to avoid
            new_head = sample_bool_matrix(self._state == FREE_SQUARE_MARK)
            player.dead(new_head)
            self._state[new_head] = self.get_head_mark(player.get_id())
        self._dead = set()

    def update_food(self):
        """
        Fills the board with food tokens.
        The amount of food on the board in a given time is specified by FOOD_N.
        """
        if len(self._food) < N_FOOD:
            new_food = sample_bool_matrix(self._state == FREE_SQUARE_MARK)
            self._food.add(new_food)
        for food in self._food:
            self._state[food] = FOOD_MARK

    def post_turn(self):
        for player in self.get_players():
            player.post_action(self)

    def update_players_records(self):
        if self._turn_number % BATCH_SIZE == 0:
            for player in self.get_players():
                player.update_records()

        if self._turn_number % (SAVE_RECORDS_BATCH_ITERATIONS * BATCH_SIZE) == 0:
            print("saving players records.")
            for player in self.get_players():
                fn = "{}_{}.pkl".format(player.get_type(), player.get_id())
                dump(player.get_records(), open(os.path.join(RECORDS_DIR, fn), "wb"))

        if PRINT_RECORDS:
            if self._turn_number % (PRINT_RECORDS_BATCH_ITERATIONS * BATCH_SIZE) == 0:
                print("---------")

                print("TOTAL - {} batches".format(int(self._turn_number / BATCH_SIZE)))
                print("{:^3s} {:^8s} {:^5s} {:^5s} {:^5s} {:^5s} {:^5s}".format("pid", "type", "s/i", "f/i", "d/i", "k/i", "l/i"))
                for pid, player in self.get_id_player_pairs():
                    records = player.get_records()
                    den = len(records["score"]) * BATCH_SIZE  # normalization factor
                    print("{:^3d} {:^8s} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                        pid,
                        player.get_type(),
                        records["score"][-1] / den,
                        records["n_food"][-1] / den,
                        records["n_died"][-1] / den,
                        records["n_killed"][-1] / den,
                        np.mean(records["loss"]) if "loss" in records else 0))

                print()

                last_n = PRINT_RECORDS_BATCH_ITERATIONS
                last_den = last_n * BATCH_SIZE  # normalization factor

                def last_n_mean(arr):
                    return (arr[-1] - (arr[-last_n-1] if len(arr) > last_n else 0)) / last_den

                print("LAST UPDATE - {} batches".format(last_n))
                print("{:^3s} {:^8s} {:^5s} {:^5s} {:^5s} {:^5s} {:^5s}".format("pid", "type", "s/i", "f/i", "d/i", "k/i", "l/i"))
                for pid, player in self.get_id_player_pairs():
                    records = player.get_records()
                    print("{:^3d} {:^8s} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                        pid,
                        player.get_type(),
                        last_n_mean(records["score"]),
                        last_n_mean(records["n_food"]),
                        last_n_mean(records["n_died"]),
                        last_n_mean(records["n_killed"]),
                        np.mean(records["loss"][-last_n:]) if "loss" in records else 0))

                print("---------")

    def __str__(self):
        """
        String representation of the current game state.
        """
        ret = ""
        ret += "{} iters\n".format(self._turn_number)
        ret += " "
        ret += "_" * self._w
        ret += '\n'
        for i in range(self._h):
            ret += "|"
            for j in range(self._w):
                if self._state[i, j] == FOOD_MARK:
                    ret += '*'
                elif self._state[i, j] == FREE_SQUARE_MARK:
                    ret += " "
                elif self._state[i, j] < 0:
                    direction = self._players_dict[int(abs(self._state[i, j]))].get_direction()
                    if direction == UP:
                        ret += "^"
                    elif direction == DOWN:
                        ret += "v"
                    elif direction == RIGHT:
                        ret += ">"
                    elif direction == LEFT:
                        ret += "<"
                else:
                    ret += str(int(self._state[i, j]))
            ret += "|"
            ret += '\n'
        ret += " "
        ret += "_" * self._w
        ret += '\n'
        for pid, player in self._players_dict.items():
            ret += " "
            t = [str(pid), player.get_type(),
                 "SCORE", str(player.get_score())]
            left_over = self._w - (sum([len(i) for i in t]) + 2)
            ret += " ".join(t[:2])
            ret += " " * left_over
            ret += " ".join(t[2:])
            ret += "\n"

        return ret

    def get_id_player_pairs(self):
        return self._players_dict.items()

    def get_ids(self):
        return self._players_dict.keys()

    def get_players(self):
        return self._players_dict.values()

    def get_player(self, pid):
        return self._players_dict[pid]

    def get_player_type(self, pid):
        return self._players_dict[pid].get_type()

    def get_food(self):
        return self._food

    def get_width(self):
        return self._w

    def get_height(self):
        return self._h

    def get_turn_number(self):
        return self._turn_number

    def get_state(self):
        """
        Returns the current state of the game.
        :return: Numpy array that describes the current state of the game.
        """
        return self._state.copy()

    def get_head_marks(self):
        return self.head_marks.copy()

    def get_body_marks(self):
        return self.body_marks.copy()

    @staticmethod
    def get_head_mark(pid):
        return -pid

    @staticmethod
    def convert_action_to_direction(action, cur_direction):
        """
        Gives an updated direction, given action and current direction
        """
        if action == FORWARD_ACTION:
            return cur_direction

        elif action == RIGHT_ACTION:
            if cur_direction == UP:
                return RIGHT
            elif cur_direction == RIGHT:
                return DOWN
            elif cur_direction == LEFT:
                return UP
            elif cur_direction == DOWN:
                return LEFT

        elif action == LEFT_ACTION:
            if cur_direction == UP:
                return LEFT
            elif cur_direction == RIGHT:
                return UP
            elif cur_direction == LEFT:
                return DOWN
            elif cur_direction == DOWN:
                return RIGHT

    @staticmethod
    def get_next_location(loc, direction, h, w):
        y, x = loc
        n_y, n_x = y, x
        if direction == UP:
            n_y = (y - 1) % h
        elif direction == DOWN:
            n_y = (y + 1) % h
        elif direction == RIGHT:
            n_x = (x + 1) % w
        elif direction == LEFT:
            n_x = (x - 1) % w
        # else:
        #     assert 0

        next_loc = n_y, n_x
        return next_loc

    @staticmethod
    def clean_records():
        for fn in os.listdir(RECORDS_DIR):
            fp = os.path.join(RECORDS_DIR, fn)
            os.unlink(fp)
