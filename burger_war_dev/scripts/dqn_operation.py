#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import subprocess
import json
import requests

import rospy
import rosparam
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from tf import transformations as tft

import cv2
import torch
import torchvision
import numpy as np
from PIL import Image as IMG
from cv_bridge import CvBridge, CvBridgeError

# config
FIELD_SCALE = 2.4
FIELD_MARKERS = {
    "Tomato_N": [(1, 8), (1, 9), (2, 8), (2, 9)],
    "Tomato_S": [(3, 6), (3, 7), (4, 6), (4, 7)],
    "Omelette_N": [(6, 13), (6, 14), (7, 13), (7, 14)],
    "Omelette_S": [(8, 11), (8, 12), (9, 11), (9, 12)],
    "Pudding_N": [(6, 3), (6, 4), (7, 3), (7, 4)],
    "Pudding_S": [(8, 1), (8, 2), (9, 1), (9, 2)],
    "OctopusWiener_N": [(11, 8), (11, 9), (12, 8), (12, 9)],
    "OctopusWiener_S": [(13, 6), (13, 7), (14, 6), (14, 7)],
    "FriedShrimp_N": [(6, 8), (6, 9), (7, 8), (7, 9)],
    "FriedShrimp_E": [(8, 8), (8, 9), (9, 8), (9, 9)],
    "FriedShrimp_W": [(6, 6), (6, 7), (7, 6), (7, 7)],
    "FriedShrimp_S": [(8, 6), (8, 7), (9, 6), (9, 7)],
}
ROBOT_MARKERS = ["BL_B", "BL_L", "BL_R", "RE_B", "RE_L", "RE_R"]

JUDGE_URL = ""


# functions
def get_rotation_matrix(rad, color='r'):
    if color == 'b' : rad += np.pi
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    return rot


def send_to_judge(url, data):
    res = requests.post(url,
                        json.dumps(data),
                        headers={'Content-Type': 'application/json'}
                        )
    return res


# main class
class DQNBot:
    """
    An operator to train the dqn agent.

    Attributes:
        lidar_ranges (tensor, (1, 360)): lidar distance data every 1 deg for 0-360 deg
        my_pose (array-like, (2, )): my robot's pose (x, y)
        image (tensor, (1, 3, 480, 640)): camera image
    """
    def __init__(self, robot="r"):
        """
        Args:
            robot ([type]): [description]
        """
        # robot attributes
        self.robot = robot
        self.my_markers = ROBOT_MARKERS[:3] if robot == "b" else ROBOT_MARKERS[3:]
        self.score = {k: 0 for k in FIELD_MARKERS.keys() + ROBOT_MARKERS}
        self.past_score = {k: 0 for k in FIELD_MARKERS.keys() + ROBOT_MARKERS}

        # state variables
        self.lidar_ranges = None
        self.my_pose = None
        self.image = None
        self.state = None
        self.past_state = None
        self.action = None

        # other variables
        self.game_state = "stop"
        self.step = 0
        self.episode = 0
        self.bridge = CvBridge()

        # rostopic subscription
        self.lidar_sub = rospy.Subscriber('scan', LaserScan, self.callback_lidar)
        self.image_sub = rospy.Subscriber('image_raw', Image, self.callback_image)
        self.amcl_sub = rospy.Subscriber("amcl_pose", PoseWithCovarianceStamped, self.callback_amcl)

        # rostopic publication
        self.twist_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        # rostopic service
        self.state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.pause_service = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_service = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    
    def callback_lidar(self, data):
        """
        callback function of lidar subscription

        Args:
            data (LaserScan): distance data of lidar
        """
        self.lidar_ranges = torch.FloatTensor(data.ranges).unsqueeze(0)

    def callback_image(self, data):
        """
        callback function of image subscription

        Args:
            data (Image): image from from camera mounted on the robot
        """
        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            img = IMG.fromarray(img)
            img = torchvision.transforms.ToTensor()(img)
            self.image = img.unsqueeze(0)
        except CvBridgeError as e:
            rospy.logerr(e)
        
    def callback_amcl(self, data):
        """
        callback function of odom subscription

        Args:
            data (PoseWithCovarianceStamped): robot pose
        """
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        self.my_pose = np.array([-y, x])

    def callback_warstate(self, data):
        """
        callback function of warstate subscription

        Args:
            data (String): json data of game state
        Notes:
            https://github.com/p-robotics-hub/burger_war_kit/blob/main/judge/README.md
        """
        json_dict = json.loads(data.data)
        self.game_state = json_dict['state']
        
        if self.game_state == "running":            
            for tg in json_dict["targets"]:
                if tg["player"] == self.robot:
                    self.score[tg["name"]] = int(tg["point"])
                else:
                    self.score[tg["name"]] = -int(tg["point"])

    def get_reward(self, past, current):
        """
        reward function.
        
        Args:
            past (dict): score dictionary at previous step
            current (dict): score dictionary at current step

        Return:
            reward (int)
        """
        diff_my_score = {k: current[k] - past[k] for k in self.score.keys() if k not in self.my_markers}
        diff_op_score = {k: current[k] - past[k] for k in self.my_markers}

        plus_diff = sum([v for v in diff_my_score.values() if v > 0])
        minus_diff = sum([v for v in diff_op_score.values() if v < 0])

        return plus_diff + minus_diff

    def get_map(self):
        
        # pose map
        rotate_matrix = get_rotation_matrix(-45 * np.pi / 180, self.robot)
        rotated_pose = np.dot(rotate_matrix, self.my_pose) / FIELD_SCALE + 0.5
        pose_map = np.zeros((16, 16))
        i = int(rotated_pose[0]*16)
        j = int(rotated_pose[1]*16)
        if i < 0: i = 0
        if i > 15: i = 15
        if j < 0: j = 0
        if j > 15: j = 15
        pose_map[i][j] = 1

        # score map
        score_map = np.zeros((16, 16))
        for key, pos in FIELD_MARKERS.items():
            for p in pos:
                score_map[p[0], p[1]] = self.score[key]

        map_array = np.stack([pose_map, score_map])

        return torch.FloatTensor(map_array).unsqueeze(0)

    def strategy(self):

        # past state
        self.past_state = self.state

        # get 2d state map
        map = self.get_map()

        # current state
        self.state = {
            "lidar": self.lidar_ranges,     # (1, 360)
            "map": map,                     # (1, 2, 16, 16)
            "image": self.image             # (1, 3, 480, 640)
        }

        if self.step != 0:
            reward = self.get_reward(self.past_score, self.score)
            reward = torch.FloatTensor(reward).unsqueeze(0)
            # push (s, s', a, r) = (self.past_state, self.state, self.action, reward) to memory


        # get action from agent
        self.action = self.agent.get_action()
        choice = int(self.action.item())

        # update twist
        twist = Twist()
        twist.linear.x = ACTION_LIST[choice][0]
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = ACTION_LIST[choice][1]
        self.twist_pub.publish(twist)

        self.past_score = self.score

        self.step += 1

    def move_robot(self, model_name, position=None, orientation=None):
        state = ModelState()
        state.model_name = model_name
        pose = Pose()
        if position is not None:
            pose.position = Point(*position)
        if orientation is not None:
            tmpq = tft.quaternion_from_euler(*orientation)
            pose.orientation = Quaternion(tmpq[0], tmpq[1], tmpq[2], tmpq[3])
        state.pose = pose
        try:
            self.state_service(state)
        except rospy.ServiceException, e:
            print("Service call failed: %s".format(e))

    def stop(self):
        self.pause_service()

    def restart(self):
        self.episode += 1

        # restart judge server
        res = send_to_judge(JUDGE_URL + "/warState/state", {"state": "running"})

        # restart gazebo physics
        self.unpause_service()

    def reset(self):
        # reset parameters
        self.step = 0
        self.score = {k: 0 for k in FIELD_MARKERS.keys() + ROBOT_MARKERS}
        self.past_score = {k: 0 for k in FIELD_MARKERS.keys() + ROBOT_MARKERS}
        self.lidar_ranges = None
        self.my_pose = None
        self.image = None
        self.state = None
        self.past_state = None
        self.action = None

        # reset judge server
        resp = requests.get(JUDGE_URL + "/reset")

        # reset robot's positions
        self.move_robot("red_bot", (0.0, -1.3, 0.0), (0, 0.0, 0))
        self.move_robot("blue_bot", (0.0, 1.3, 0.0), (0, 0.0, 0))

    def train(self, n_epochs=20):
        for _ in range(n_epochs):
            pass
    
    def run(self, rospy_rate=1):

        r = rospy.Rate(rospy_rate)

        while not rospy.is_shutdown():
            
            while not all([v is not None for v in [self.lidar_ranges, self.my_pose, self.image]]):
                pass

            if self.game_state == "stop":
                
                # stop the game
                self.stop()

                # train agent
                self.train(n_epochs=20)

                # update target q function
                if self.episode % UPDATE_Q_FREQ == 0:
                    pass

                # reset the game
                self.reset()

                # restart the game
                self.restart()

            else:
                self.strategy()

            r.sleep()

    
if __name__ == "__main__":
    rospy.init_node('dqn_run')
    JUDGE_URL = rospy.get_param('/send_id_to_judge/judge_url')

    # parameters
    VEL = 0.4
    OMEGA = 1
    ACTION_LIST = [
        [0, 0],
        [VEL, 0],
        [-VEL, 0],
        [0, OMEGA],
        [0, -OMEGA],
    ]
    UPDATE_Q_FREQ = 5

    try:
        bot = DQNBot()
        bot.run(1)

    except rospy.ROSInterruptException:
        pass
