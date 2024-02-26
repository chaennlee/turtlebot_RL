#!/usr/bin/python3

import torch
import time
import os
import rospy


class SaveData:
    def __init__(self, stage):
        self.current_time = time.localtime()
        self.year = self.current_time.tm_year
        self.month = self.current_time.tm_mon
        self.day = self.current_time.tm_mday
        self.hour = self.current_time.tm_hour
        self.minute = self.current_time.tm_min
        self.second = self.current_time.tm_sec
        self.h = self.current_time.tm_hour
        self.m = self.current_time.tm_min
        self.s = self.current_time.tm_sec
        self.stage = stage

        self.dirPath = os.path.dirname(os.path.realpath(__file__))

    def makeDir(self):
        # self.dirPath = os.path.dirname(os.path.realpath(__file__))
        dirPath = self.dirPath.replace(
            "dqn_ttb/src/turtlebot3_dqn", "dqn_ttb/save_model"
        )
        dirPath = dirPath + "/stage_{}/{}/{:02d}/{:02d} {}:{}:{}".format(
            self.stage,
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
        )
        self.dirPath = dirPath

        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
            rospy.loginfo("Directory for save is created!!")
        else:
            rospy.loginfo("Directory already exists!!")

        ## 나머지 파일도 만들어준다.
        self.dirPolicy = dirPath + "/policy_net"
        self.dirPerformance = dirPath + "/performance"
        self.dirMemory = dirPath + "/memory"
        os.makedirs(self.dirPolicy)
        os.makedirs(self.dirPerformance)
        os.makedirs(self.dirMemory)

    def memorySavedAt(self):
        return self.dirMemory

    def saveModel(self, model, episode, optimizer, score):
        dir_path = self.dirPolicy
        save_path = (
            dir_path
            + "/stage_"
            + str(self.stage)
            + "_policy_net_"
            + str(episode)
            + ".tar"
        )

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            rospy.loginfo("Directory for save is created!!")

        # rospy.loginfo("save_path: {}".format(save_path))

        torch.save(
            {
                "epoch": episode,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "score": score,
                # 'loss' : loss
            },
            save_path,
        )
        rospy.loginfo("policy save success @episode %d", episode)

    def recordPerformance(
        self,
        episode,
        score,
        lengthOfMemory,
        epsilon,
        loss_item,
        h,
        m,
        s,
        start_time,
        total_step,
    ):
        dir_path = self.dirPerformance
        file_name = (
            dir_path
            + "/stage_"
            + str(self.stage)
            + "_record_performance@"
            + str(start_time)
            + ".txt"
        )

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            rospy.loginfo("Directory for performance is now created!!")

        with open(file_name, "a") as outfile:
            outfile.write(
                "episode: {}, score: {}, memory: {}, epsilon: {}, loss: {}, step: {}, time: {}:{}:{}\n".format(
                    episode,
                    score,
                    lengthOfMemory,
                    epsilon,
                    loss_item,
                    total_step,
                    h,
                    m,
                    s,
                )
            )
            rospy.loginfo("record data success @episode %d", episode)
            # rospy.loginfo("performance directory : %s",file_name)

    def loadModel(self, episode, file_name, policy_net, optimizer, evaluation):
        folder = os.path.dirname(os.path.realpath(__file__))
        folder = folder.replace("dqn_ttb/src/turtlebot3_dqn", "dqn_ttb/nodes/")
        load_path = folder + file_name + ".tar"

        checkout = torch.load(load_path)
        load_episode = checkout["epoch"]
        policy_net.load_state_dict(checkout["model_state_dict"])
        optimizer.load_state_dict(checkout["optimizer_state_dict"])
        score = checkout["score"]

        if evaluation:
            policy_net.eval()
        else:
            policy_net.train()

        rospy.loginfo("torch load success @episode %d", episode)
        return load_episode, policy_net, optimizer, score
