from constants import *
from config import *


class BasePlayer:
    """
    Player object that the game holds for each player.
    """

    def __init__(self, pid, head, leftover=STARTING_LENGTH):
        """
        Initiates a new GamePlayer object, identified by pid
        :param pid: Player identifier
        """
        self.locations_set = set()
        self.head = head
        self.locations = [self.head]
        self.tail = self.locations[-1]
        self.direction = UP
        self.pid = pid
        self.score = 0
        self.leftover_counter = leftover

    @staticmethod
    def get_type():
        pass

    # virtual
    def pre_action(self, game):
        pass

    # virtual
    def get_action(self, game):
        pass

    # virtual
    def post_action(self, game):
        pass

    # probably not necessary
    # def init_snake(self):
    #     """
    #     Initializes the starting position of the snake.
    #     """
    #     counter = 1
    #     hx, hy = self.head
    #     while counter < STARTING_LENGTH:
    #         self.locations.append((hx + counter, hy))
    #         counter += 1
    #     self.locations_set = set(self.locations)
    #     self.locations = [self.head] + self.locations

    def move(self, new_pos):
        """
        Advances the snake by one move.
        :param new_pos: New location of the head.
        """
        # TODO
        # check if valid position

        if self.leftover_counter == 0:
            self.locations.pop()

        else:
            self.leftover_counter -= 1

        self.locations_set = set(self.locations)

        self.locations = [new_pos] + self.locations

        self.set_head(new_pos)
        self.set_tail(self.locations[-1])

    def get_location_set(self):
        """
        :return: The set of locations of the snake, without the head.
        """
        return self.locations_set

    def get_locations(self):
        """
        :return: THe list of locations of the snake.
        """
        return self.locations

    def update_score(self, to_add):
        """
        Adds to_add to the score of the snake.
        :param to_add: The score to add.
        """
        self.score += to_add

    def set_head(self, new_head):
        """
        Set the head of the snake to be at the specified location
        :param new_head: New location of the head.
        """
        self.head = new_head

    def get_head(self):
        """
        :return: The coordinates of the snakes head.
        """
        return self.head

    def set_tail(self, new_tail):
        """
        Set the tail of the snake to be at the specified location
        :param new_tail: New location of the tail.
        """
        self.tail = new_tail

    def get_tail(self):
        """
        :return: The coordinates of the snakes head.
        """
        return self.tail

    # def alive(self):
    #     """
    #     Is the snake alive?
    #     :return: True if yes, False otherwise.
    #     """
    #     return self.is_alive

    def dead(self, new_head):
        """
        Pronounce the snake dead.
        """
        self.score = 0
        self.head = new_head
        self.locations = [self.head]
        self.leftover_counter = STARTING_LENGTH

    def get_score(self):
        """
        :return: The score of the snake.
        """
        return self.score

    def get_direction(self):
        """
        :return: The current direction of the snake.
        """
        return self.direction

    def set_direction(self, direction):
        """
        Set the direction of the snake.
        :param direction: UP, RIGHT, LEFT or DOWN
        """
        self.direction = direction

    def get_id(self):
        """
        :return: The id of the snake.
        """
        return self.pid

    def update_leftover(self, n):
        """
        Add n to the length that the snake is yet to grow
        :param n:
        :return:
        """
        self.leftover_counter += n

    # def check_closed(self):
    #     """
    #
    #     :return:
    #     """
    #     return self.head in self.locations_set
    #
    # def get_enclosure(self):
    #     """
    #
    #     :return:
    #     """
    #     pass
