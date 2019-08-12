from keras import Sequential
from keras.layers import Convolution2D, Activation, Dense, Flatten
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np
import time
import os

from players.base_player import BasePlayer
from constants import *
from config import *
from utils import get_greedy_action

DIRECTION_TO_N_ROT90 = {
    UP: 0,
    RIGHT: 1,
    DOWN: 2,
    LEFT: 3
}


class CNNPlayer(BasePlayer):
    def __init__(self, pid, head):
        super().__init__(pid, head)
        self.input_shape = (GAME_HEIGHT, GAME_WIDTH, N_INPUT_CHANNELS)
        self.center_y = GAME_HEIGHT // 2
        self.center_x = GAME_WIDTH // 2
        self.model = self.build_model()
        self.prev_state = -1
        self.prev_score = -1
        self.prev_q_values = -1
        self.action_index = -1
        self.batch = []
        self.n_batches = 0

        if LOAD_MODEL:
            print("loading model: {}".format(LOAD_MODEL_FILE_NAME))
            self.model = load_model(os.path.join(MODELS_DIR, LOAD_MODEL_FILE_NAME))

    @staticmethod
    def get_type():
        return CNN_PLAYER

    def pre_action(self, game):
        self.prev_state = self.extract_model_input(game)
        self.prev_score = self.get_score()

    def get_action(self, game):
        q_values = self.model.predict(self.prev_state)
        self.prev_q_values = q_values

        rand = np.random.random()
        if rand < EPSILON_GREEDY:
            action_index = np.random.randint(N_ACTIONS)
        else:
            action_index = np.argmax(q_values)
        self.action_index = action_index
        action = ACTIONS[action_index]
        return action

        # # todo tmp
        # greedy_action = get_greedy_action(game, self.head, self.direction)
        # return greedy_action

    def post_action(self, game):
        if not TRAIN_MODEL:
            return

        cur_state = self.extract_model_input(game)
        reward = self.get_score() - self.prev_score
        sample = (self.prev_state, self.action_index, reward, cur_state)
        self.batch.append(sample)

        if len(self.batch) == BATCH_SIZE:
            x = np.zeros((BATCH_SIZE,) + self.input_shape)
            y = np.zeros((BATCH_SIZE, N_ACTIONS))

            for i in range(len(self.batch)):
                state_t, action_index, reward, state_t1 = self.batch[i]
                q_values_t1 = self.model.predict(state_t1)

                x[i] = state_t
                y[i] = self.prev_q_values  # loss is affected only by action_index value
                y[i, action_index] = reward + GAMMA * np.max(q_values_t1)

            loss = self.model.train_on_batch(x, y)
            self.batch = []
            self.n_batches += 1

            if self.n_batches % PRINT_LOSS_BATCH_ITERATIONS == 0:
                print("loss = {:.3f}".format(loss))

            if self.n_batches % SCORE_SUMMARY_BATCH_ITERATION == 0:
                print("---------")
                print("{} iters, {} batches".format(game.get_turn_number(), self.n_batches))
                n = game.get_turn_number()
                for pid, player in game.get_id_player_pairs():
                    print("{} {:5s} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                        pid,
                        player.get_type(),
                        player.score / n,
                        player.n_food_eaten / n,
                        player.n_died / n,
                        player.n_killed / n))
                print("---------")

            if SAVE_MODEL:
                if self.n_batches % SAVE_MODEL_BATCH_ITERATIONS == 0:
                    # todo tmp
                    print("saving model: {}".format(time.strftime("%Y-%m-%d-%H-%M-%S")))  # todo rm
                    model_fn = "{}.h5".format(time.strftime("%Y-%m-%d-%H-%M-%S"))
                    self.model.save(os.path.join(MODELS_DIR, model_fn))

    # CNN impl.
    def build_model(self):
        model = Sequential()
        model.add(Convolution2D(1, (2, 2), strides=(1, 1), input_shape=self.input_shape))
        model.add(Activation("relu"))
        # model.add(Convolution2D(4, (2, 2), strides=(1, 1)))
        # model.add(Activation("relu"))
        model.add(Flatten())
        # model.add(Dense(8))
        model.add(Dense(N_ACTIONS))
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss="mean_squared_error", optimizer=adam)
        # print(model.summary())  # todo
        return model

    def normalize_state(self, game):
        # align state
        n_rot90 = DIRECTION_TO_N_ROT90[self.direction]
        aligned_state = np.rot90(game.get_state(), n_rot90)

        # roll s.t. head is in center
        y, x = np.where(aligned_state == game.get_head_mark(self.pid))
        assert x.shape == y.shape == (1,)
        head_y = y[0]
        head_x = x[0]
        norm_state = np.roll(np.roll(aligned_state, self.center_y-head_y, axis=0), self.center_x-head_x, axis=1)
        return norm_state

    def extract_model_input(self, game):
        # aligning and centering head
        norm_state = self.normalize_state(game)

        # head isn't modeled since it's centered
        model_input = np.zeros(self.input_shape)
        model_input[:, :, 0] = norm_state == FOOD_MARK  # food
        model_input[:, :, 1] = norm_state == self.pid  # self body
        # model_input[:, :, 2] = (norm_state != FREE_SQUARE_MARK) & \
        #                        (norm_state != FOOD_MARK) & \
        #                        (norm_state != self.pid) & \
        #                        (norm_state != game.get_head_mark(self.pid))  # other players
        model_input = model_input[np.newaxis, :]
        return model_input



# def stack_image(game_image):
#     #Make image black and white
#     x_t = skimage.color.rgb2gray(game_image)
#     #Resize the image to 80x80 pixels
#     x_t = skimage.transform.resize(x_t,(80,80))
#     #Change the intensity of colors, maximizing the intensities.
#     x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))
#     # Stacking 2 images for the agent to get understanding of speed
#     s_t = np.stack((x_t,x_t),axis=2)
#     # Reshape to make keras like it
#     s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
#     return s_t
#
# def train_network(model):
#     game_state = game.Game() #Starting up a game
#     game_state.set_start_state()
#     game_image,score,game_lost = game_state.run(0) #The game is started but no action is performed
#     s_t = stack_image(game_image)
#     terminal = False
#     t = 0
#     d = []
#     nb_epoch = 0
#     while(True):
#         loss = 0
#         Q_sa = 0
#         action_index = 4
#         r_t = 0
#         a_t = 'no nothing'
#         if terminal:
#             game_state.set_start_state()
#         if t % NB_FRAMES == 0:
#             if random.random() <= EPSILON:
#                 action_index = random.randrange(NB_ACTIONS)
#                 a_t = GAME_INPUT[action_index]
#             else:
#                 action_index = np.argmax(model.predict(s_t))
#                 a_t = GAME_INPUT[action_index]
#         #run the selected action and observed next state and reward
# 	    x_t1_colored, r_t, terminal = game_state.run(a_t)
# 	    s_t1 = stack_image(x_t1_colored)
# 	    d.append((s_t, a_t, r_t, s_t1))
#
# if len(d)==BATCH:
# 	        inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
# 	        targets = np.zeros((BATCH, NB_ACTIONS))
# 	        i = 0
# 	        for s,a,r,s_pred in d:
# 	            inputs[i:i + 1] = s
# 	            if r < 0:
# 	                targets[i ,a] = r
# 	            else:
# 	                Q_sa = model.predict(s_pred)
# 	                targets[i ,a] = r + GAMMA * np.max(Q_sa)
# 	            i+=1
# 	        loss += model.train_on_batch(inputs,targets)
# 	        d.clear()
# 	        #Exploration vs Exploitation
# 	        if EPSILON > FINAL_EPSILON:
# 	            EPSILON -= EPSILON/500
