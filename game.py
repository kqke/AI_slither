# TODO
# replace game_player by base_player - need pid in cnn_player (more intuitive implementation by using inheritance)
# kepp a direction member
# remove p.get_id() - 1 from code - prone to bugs (replace by dict or function if still necessary)
# food should be represented in state
# x corresponds to width and y to height (stick with this convention - o.w. prone to bugs)
# search for todo s ...
# if reaching end state - don't update cnn player

import numpy as np
from players.cnn_player import CNNPlayer
from players.greedy_player import GreedyPlayer
from players.random_player import RandomPlayer
from players.manual_player import ManualPlayer
from constants import *
from config import *
from time import sleep
from os import system, name


SCORE_MULTIPLIER = 2
DIRECTION = 6
PLAYER = 5
HEAD_POS = 4
LOCATION = 3
TAIL_POS = 2
IN_PLAY = 1
SCORE = 0


def clear():
    system("cls" if name == "nt" else "clear")


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
        self._h, self._w = height, width
        self._state = np.zeros((height, width))
        self._check = self._state.copy()
        self._players_dict = dict()
        self._food = set()
        self._turn_number = 0
        self._dead = []
        self.init_players(players)
        self.update_food()
        self.update_board()

    def init_players(self, players):
        """
        Factory method that initializes player instances.
        :param players: A dict that contains key-value pairs thar correspond to player type and their amount.
        """
        pid = 1
        for player in players:
            n = players[player]
            for k in range(n):
                head = np.random.choice(np.where(self._state == 0))
                if player == CNN_PLAYER:
                    self._players_dict[pid] = CNNPlayer(pid, head)
                elif player == GREEDY_PLAYER:
                    self._players_dict[pid] = GreedyPlayer(pid, head)
                elif player == RANDOM_PLAYER:
                    self._players_dict[pid] = RandomPlayer(pid, head)
                elif player == MANUAL_PLAYER:
                    self._players_dict[pid] = ManualPlayer(pid, head)
                else:
                    assert 0
                pid += 1

    def get_players(self):
        return self._players_dict.values()

    # TODO
    # different kinds of food, eg. different scores, different resulting snake growth
    def update_food(self):
        """
        Fills the board with food tokens.
        The amount of food on the board in a given time is specified by FOOD_N.
        """
        if len(self._food) < FOOD_N:
            new_food = np.random.choice(np.where(self._state == 0))
            self._food.add(new_food)
        for food in self._food:
            self._state[food] = FOOD

    def run(self, turns):
        """
        Runs the game for max_turns (specified in constructor) turns.
        """
        while self._turn_number < turns:
            if DISPLAY:
                print(self)
                sleep(RENDER_DELAY)
                # clear()  # todo
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
        self._turn_number += 1

    def pre_turn(self):
        for player in self.get_players():
            player.pre_action(self)

    def move_players(self):
        """
        Move each player to its next position.
        """
        for player in self.get_players():
            self.do_action(player, self.check_food(player))

    def check_food(self, player):
        """
        Checks whether a snake has eaten a food token in the previous turn,
        if so, its score is updated, and true is returned.
        :return:
        """
        if player.get_head() in self._food:
            player.update_score(FOOD_PRIZE)
            self._food.remove(player.get_head())
            return 1
        return 0

    def check_collisions(self):
        """
        Checks whether two snakes have collided, if so, the colliding snake is pronounced dead.
        In head-on collision, the longer snake wins.
        """
        for p1 in self.get_players():
            if p1.get_head() in p1.get_location_set():
                self._dead.append(p1)
                break
            for p2 in self.get_players():
                if p1 is not p2:
                    if p2.alive():
                        if p1.get_head() in p2.get_location_set():
                            self._dead.append(p1)
                            p2.update_score(p1.get_score())
                        elif p1.get_head() == p2.get_head():
                            # todo
                            # in the case of head on collision of snakes of the same length:
                            # currently an arbitrarily chosen snake dies
                            smaller = p1 if len(p1.get_locations()) > len(p2.get_locations()) else p2
                            other = p1 if smaller == p2 else p2
                            self._dead.append(smaller)
                            other.update_score(smaller.get_score())

    def update_board(self):
        """
        Generates a numpy array corresponding to the current game state.
        """
        self._state = np.zeros((self._h, self._w))
        for pid, player in self._players_dict.items():
            if player not in self._dead:
                for pos in player.get_locations():
                    self._state[pos] = pid
                self._state[player.get_head()] = -pid
        self.update_dead()

    def update_dead(self):
        for dead in self._dead:
            new_head = np.random.choice(np.where(self._state == 0))
            dead.dead(new_head)
            self._state[new_head] = - dead.get_id()
        self._dead = []

    def post_turn(self):
        for player in self.get_players():
            player.post_action(self)

    def do_action(self, player, food):
        """
        Advances a player according to its action.
        :param player: The player.
        :param food: Whether it got a food token in the previous round.
        """
        action = player.get_action(self)

        direction = self.convert_action_to_direction(action, player.get_direction())
        player.set_direction(direction)
        x, y = player.get_head()
        n_x, n_y = x, y
        if direction == UP:
            n_x = (x - 1) % self._h
        elif direction == DOWN:
            n_x = (x + 1) % self._h
        elif direction == RIGHT:
            n_y = (y + 1) % self._w
        elif direction == LEFT:
            n_y = (y - 1) % self._w
        player.move((n_x, n_y), food)

    def get_state(self):
        """
        Returns the current state of the game.
        :return: Numpy array that describes the current state of the game.
        """
        return self._state.copy()

    def score_func(self):
        """
        Barak?
        """
        return self._turn_number * SCORE_MULTIPLIER

    def __str__(self):
        """
        String representation of the current game state.
        """
        ret = " "
        ret += "_" * self._w
        ret += '\n'
        for i in range(self._h):
            ret += "|"
            for j in range(self._w):
                if self._state[i, j] == 100:
                    ret += '*'
                elif self._state[i, j] == 0:
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
                 SCORE_STR, str(player.get_score())]
            left_over = self._w - (sum([len(i) for i in t]) + 2)
            ret += " ".join(t[:2])
            ret += " " * left_over
            ret += " ".join(t[2:])
            ret += "\n"

        return ret


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

    # def check_enclosure(self):
    #     """
    #     Checks whether a snake has been caught in the enclosure of another snake, if so,
    #     the enclosed snake is pronounced dead.
    #     """
    #     enclosures = []
    #     for player in self._game_players:
    #         if player.check_closed():
    #             enclosures.append(player.get_enclosure())
    #     for enclosure in enclosures:
    #         for player in self._game_players:
    #             if self.check_in_enclosure(player.get_head(), enclosure):
    #                 player.dead()
    # circles_to_check = []
    # for player in self._game_players:
    #     if player_info[IN_PLAY]:
    #         self.do_action(player, player.get_action(self))
    #         closed, pos = self.check_closed(player_info[HEAD_POS], player_info[TAIL_POS], player_info[DIRECTION])
    #         if closed:
    #             circles_to_check.append(set(player_info[LOCATION][0:player_info[LOCATION].index(pos)]))

    # def check_in_enclosure(self, point, enclosure):
    #     """
    #
    #     :param point:
    #     :param enclosure:
    #     :return:
    #     """
    #     pass

    # def check_closed(self, head_pos, tail_pos, direction):
    #     # not sure if necessary
    #     hx = head_pos[0]
    #     tx = tail_pos[0]
    #     ty = tail_pos[1]
    #     hy = head_pos[1]
    #     xs = [-1, 0, 1]
    #     ys = [-1, 0, 1]
    #     count = 0
    #     for x in xs:
    #         if direction == RIGHT:
    #             xs.remove(-1)
    #         elif direction == LEFT:
    #             xs.remove(1)
    #
    #         for y in ys:
    #             if direction == UP:
    #                 xs.remove(-1)
    #             elif direction == DOWN:
    #                 xs.remove(1)
    #
    #             if y == 0 and x == 0:
    #                 continue
    #             elif y == ty and x == tx:
    #                 if not(self._state[hx, hy] >= 100):
    #                     continue
    #
    #             if self._state[hx + x, hy + y] == self._state[hx, hy]:
    #                     return True, [x, y]
    #     return False, []
