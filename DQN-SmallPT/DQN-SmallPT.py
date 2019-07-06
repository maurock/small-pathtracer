import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
import random
import gym
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Variables for tuning
N_EPISODES = 1000
state_space = 112
action_space = 7
training_mode = True  # True if training phase, False if test phase


class DQN:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = 0.001
        self.minibatch_size = 64
        self.memory = []
        self.gamma = 0.9  # discount rate
        self.exploration_rate = 1  # exploration rate
        self.epsilon_decay = 0.955
        self.epsilon_min = 0.01
        self.model = self.network()
        self.memory_length = 3000

    # Defining DQN.
    # properties of DQN ------------------------------
    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(output_dim=100, activation='tanh', input_dim=self.state_space))
        model.add(Dense(output_dim=100, activation='tanh'))
        model.add(Dense(output_dim=self.action_space, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
            print("weights loaded")
        return model

    # Predict action based on current state
    def do_action(self, state):
        # TODO: convert from one-hot to continuous
        action = np.zeros(7)
        if np.random.rand() <= self.exploration_rate:
            action[np.random.randint(0, 6)] = 1
            arr = "RANDOM"
            return action, arr
        arr = self.model.predict(state.reshape(1, state_space))
        action[np.argmax(arr[0])] = 1
        arr = arr.reshape(7)
        return action, arr

    # Store info for replay memory
    # store in global variable memory and returns it
    def remember(self, state, action, next_state, reward, done):  # not consider: timeout, grass.
        if ((len(self.memory) + 1) > self.memory_length):
            del self.memory[0]
        self.memory.append([state, action, next_state, reward, done])

    def train(self):
        if len(self.memory) > self.minibatch_size:
            minibatch = random.sample(self.memory, self.minibatch_size)
        else:
            minibatch = self.memory

        for state, action, next_state, reward, done in minibatch:
            target = reward

            if done != 5:
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, state_space)))
            target_f = self.model.predict(state.reshape(1, state_space))
            target_f[0][np.argmax(action)] = target
            self.model.fit(state.reshape(1, state_space), target_f, epochs=1, verbose=0)

        # if self.exploration_rate > self.epsilon_min:
        #   self.exploration_rate *= self.epsilon_decay

    def get_reward_perfect(self, done, state):
        state_reward = state[:-2]
        quadratic_deviation = state[-2]
        state_reward = np.resize(state_reward, (10, 11))
        reward = 0

        # LIVE REWARD ---------
        for i in range(10):  # negative reward linear to occupancy along the cells. worst case: -27.5
            crash_p_car = state_reward[i][0]  # check probability of car occupancy
            crash_p_bike = state_reward[i][6]  # check probability of bike occupancy

            reward += -(5 - 0.5 * i) * crash_p_car  # check probability of car occupancy
            reward += -(5 - 0.5 * i) * crash_p_bike  # check probability of bike occupancy
            reward += -quadratic_deviation  # check quadratic deviation, already normalized

        # FINAL REWARD ---------
        if done == 5:
            reward += - 100  # if high occupancy probability

        return reward

    def get_reward_exp1(self, done, state, row):
        '''
        Reward function designed by the expert
        :param crash: true if ego crashed, false otherwise   ------------- done[2]
        :param state: feature space wrt ego
        :param row: list of vehicles and bikes with row wrt ego
        :return: cumulative reward considering its various components based on the state
        '''

        arr_reward = np.zeros(5)
        for i in range(3):
            crash_p_car = state[i][0]  # check probability of occupancy in first 3 cells
            if (crash_p_car > 0.5):
                arr_reward[0] = 1
                break
            crash_p_bike = state[i][8]  # check probability of occupancy in first 3 cells
            if (crash_p_bike > 0.5):
                arr_reward[0] = 1
                break

        for i in range(3):
            right_of_way_car = len(row[i][0])  # check if car objects had right of way in first 3 cells
            right_of_way_bike = len(row[i][2])  # check if car objects had right of way in first 3 cells

            if (right_of_way_car or right_of_way_bike):
                arr_reward[1] = 1
                break

        for i in range(3):
            is_crossing = state[i][17]
            if (is_crossing):  # check if ego crashes in crossing path
                arr_reward[2] = 1
                break

        for i in range(3):
            crash_p_car = state[i][0]  # check probability of occupancy in first 3 cells
            if (crash_p_car > 0.5):
                arr_reward[3] = 1
                break
            crash_p_bike = state[i][8]  # check probability of occupancy in first 3 cells
            if (crash_p_bike > 0.5):
                arr_reward[3] = 1
                break

        for i in range(3):
            right_of_way_car = len(row[i][0])  # check if car objects had right of way in first 3 cells
            right_of_way_bike = len(row[i][2])  # check if car objects had right of way in first 3 cells

            if (right_of_way_car or right_of_way_bike):
                arr_reward[4] = 1
                break
        reward = 0
        if done == 5:  # crash with ego
            if (arr_reward[0]):
                reward = reward - 2  # if high occupancy probability
            if (arr_reward[1]):
                reward = reward - 4  # if actors had row
            if (arr_reward[2]):
                reward = reward - 5  # if ego in crossing path
        else:
            if (arr_reward[0]):
                reward = reward + 2  # if high occupancy probability
            if (arr_reward[1]):
                reward = reward + 1  # if actors had row

        return reward


def discretize(element, n_bin, minimum, maximum):
    try:
        array = np.zeros(n_bin)
        if element == maximum:
            pos = n_bin - 1
        else:
            interval = (maximum - minimum) / n_bin
            pos = math.floor(element / interval)
        array[pos] = 1
        return array
    except IndexError:
        print("Element outside the range")


# Main
if __name__ == "__main__":
    state_space = 5
    action_space = 72
    agent = DQN(state_space, action_space)
    model = agent.network()
    score = []
    episode = []
    episode_temp = 0
    while True:

        # Get state from C++
        f = open("state.txt", "r")
        lines = f.readlines()
        if (len(lines) > 1):
            while lines[0] != "1\n":
                f = open("../state.txt", "r")
                lines = f.readlines()
            print(state)
            f.close()
            state = np.array([])
            for i in lines.split(','):
                state = np.append(i)
            f = open('state.txt', "w")
            string = '0'
            f.write(string)
            f.close()
            done = False

            while not done:

                # Predict action
                action = agent.do_action(state)

                # Send action to C++
                f = open("../action.txt", "r")
                lines = f.readlines()
                print("fino a qui tutto bene2")
                while lines[0] != "1\n":
                    f = open("python-write.txt", "r")
                    lines = f.readlines()
                print("fino a qui tutto bene3")
                f.close()
                f = open('../action.txt', "w")
                f.write('0\n')
                f.write(action)
                f.close()

                # Read SARS'A
                f = open("../sarsa.txt", "r")
                lines = f.readlines()
                if (len(lines) > 1):
                    while lines[0] != "1\n":
                        f = open("../sarsa.txt", "r")
                        lines = f.readlines()
                    print("fino a qui tutto bene4")
                    f.close()
                    reward = lines[1]
                    next_state = lines[2]
                    f = open('../sarsa.txt', "w")
                    string = '0'
                    f.write(string)
                    f.close()

                    if reward > 5:
                        done = True

                    # Store in memory
                    agent.remember(state, action, reward, next_state, done)

                agent.train()


