#!/usr/bin/python3

import random
from collections import deque, namedtuple
import torch
import rospy
import os
import pickle

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def __len__(self):  # 매직 매소드라고 하는가보다.
        return len(self.memory)

    def pop(self):
        return self.memory.pop()

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # def print(self):
    #     return self.memory.state

    def save(self, dir_path, episode):
        save_path = dir_path + "/memory" + str(episode) + ".pkl"

        with open(save_path, "wb") as outfile:
            pickle.dump(self.memory, outfile)

        rospy.loginfo("[MEMORY] Memory Save Success @%d", episode)
        rospy.loginfo(
            "[MEMORY] Memory Length: %d, Capacity: %d", len(self.memory), self.capacity
        )

    def load(self, file_name):
        folder = os.path.dirname(os.path.realpath(__file__))
        folder = folder.replace("dqn_ttb/src/turtlebot3_dqn", "dqn_ttb/nodes/")
        load_path = folder + file_name + ".pkl"

        with open(load_path, "rb") as infile:
            self.memory = pickle.load(infile)

        rospy.loginfo("[MEMORY] Memory Load Success")
