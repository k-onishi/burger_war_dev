#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import subprocess
import json
import requests
import copy

import rospy
import rosparam
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
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

from state import State
from agent import Agent


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
        lidar_ranges (tensor, (1, 1, 360)): lidar distance data every 1 deg for 0-360 deg
        my_pose (array-like, (2, )): my robot's pose (x, y)
        image (tensor, (1, 3, 480, 640)): camera image
    """
    def __init__(self, robot="r", online=False, policy_mode="epsilon", debug=True, save_path=None, load_path=None, manual_avoid=False):
        """
        Args:
            robot ([type]): [description]
        """
        # attributes
        self.robot = robot
        self.enemy = "b" if robot == "r" else "r"
        self.online = online
        self.policy_mode = policy_mode
        self.debug = debug
        self.my_markers = ROBOT_MARKERS[:3] if robot == "b" else ROBOT_MARKERS[3:]
        self.score = {k: 0 for k in FIELD_MARKERS.keys() + ROBOT_MARKERS}
        self.past_score = {k: 0 for k in FIELD_MARKERS.keys() + ROBOT_MARKERS}

        if save_path is None:
            self.save_path = "../catkin_ws/src/burger_war_dev/burger_war_dev/scripts/models/tmp.pth"
        else:
            self.save_path = save_path

        # state variables
        self.lidar_ranges = None
        self.my_pose = None
        self.image = None
        self.state = None
        self.past_state = None
        self.action = None

        # other variables
        self.game_state = "end"
        self.step = 0
        self.episode = 0
        self.bridge = CvBridge()

        # rostopic subscription
        self.lidar_sub = rospy.Subscriber('scan', LaserScan, self.callback_lidar)
        self.image_sub = rospy.Subscriber('image_raw', Image, self.callback_image)
        self.state_sub = rospy.Timer(rospy.Duration(0.5), self.callback_warstate)

        if self.debug:
            if self.robot == "r": self.odom_sub = rospy.Subscriber("red_bot/tracker", Odometry, self.callback_odom)
            if self.robot == "b": self.odom_sub = rospy.Subscriber("enemy_bot/tracker", Odometry, self.callback_odom)
        else:
            self.amcl_sub = rospy.Subscriber("amcl_pose", PoseWithCovarianceStamped, self.callback_amcl)

        # rostopic publication
        self.twist_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        # rostopic service
        self.state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.pause_service = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_service = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

        # agent
        self.agent = Agent(num_actions=len(ACTION_LIST), batch_size=BATCH_SIZE, capacity=MEM_CAPACITY, gamma=GAMMA, prioritized=PRIOTIZED, lr=LR)

        if load_path is not None:
            self.agent.load(load_path)

        # mode
        self.punish_if_facing_wall = not manual_avoid
    
    def callback_lidar(self, data):
        """
        callback function of lidar subscription

        Args:
            data (LaserScan): distance data of lidar
        """
        raw_lidar = data.ranges
        raw_lidar = [0.12 if l > 3.5 else l for l in raw_lidar]
        self.lidar_ranges = torch.FloatTensor(raw_lidar).view(1, 1, -1)   # (1, 1, 360)

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
            self.image = img.unsqueeze(0)                   # (1, 3, 480, 640)
        except CvBridgeError as e:
            rospy.logerr(e)
    
    def callback_odom(self, data):
        """
        callback function of tracker subscription

        Args:
            data (Odometry): robot pose
        """
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        self.my_pose = np.array([x, y])

    def callback_amcl(self, data):
        """
        callback function of amcl subscription

        Args:
            data (PoseWithCovarianceStamped): robot pose
        """
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        self.my_pose = np.array([-y, x])

    def callback_warstate(self, event):
        """
        callback function of warstate subscription

        Notes:
            https://github.com/p-robotics-hub/burger_war_kit/blob/main/judge/README.md
        """
        # get the game state from judge server by HTTP request
        resp = requests.get(JUDGE_URL + "/warState")
        json_dict = json.loads(resp.text)
        self.game_state = json_dict['state']
        
        if self.game_state == "running":            
            for tg in json_dict["targets"]:
                if tg["player"] == self.robot:
                    self.score[tg["name"]] = int(tg["point"])
                elif tg["player"] == self.enemy:
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

        # Check LiDAR data to punish for AMCL failure
        bad_position = 0
        if self.punish_if_facing_wall and (self.lidar_ranges is not None):
            lidar_1d = self.lidar_ranges.squeeze()
            count_too_close = 0
            
            # Check each laser and count up if too close
            for intensity in lidar_1d:
                if intensity.item() < DIST_TO_WALL_TH:
                    count_too_close += 1

            # Punish if too many lasers close to obstacle
            if count_too_close > NUM_LASER_CLOSE_TO_WALL_TH:
                print("### Too close to the wall, get penalty ###")
                bad_position = -1

        plus_diff = sum([v for v in diff_my_score.values() if v > 0])
        minus_diff = sum([v for v in diff_op_score.values() if v < 0])

        return plus_diff + minus_diff + bad_position

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
        self.state = State(
            self.lidar_ranges,     # (1, 1, 360)
            map,                   # (1, 2, 16, 16)
            self.image             # (1, 3, 480, 640)
        )

        if self.action is not None:
            current_score = copy.deepcopy(self.score)
            reward = self.get_reward(self.past_score, current_score)
            print("reward: {}".format(reward))
            self.past_score = current_score
            reward = torch.LongTensor([reward])
            self.agent.memorize(self.past_state, self.action, self.state, reward)

        # manual wall avoidance
        if not self.punish_if_facing_wall:
            avoid, linear_x, angular_z = self.avoid_wall()
        else:
            avoid = False

        if avoid:
            self.action = None
        else:
            # get action from agent
            self.action = self.agent.get_action(self.state, self.episode, self.policy_mode, self.debug)
            choice = int(self.action.item())

            linear_x = ACTION_LIST[choice][0]
            angular_z = ACTION_LIST[choice][1]

        print("step: {}, vel:{}, omega:{}".format(self.step, linear_x, angular_z))

        # update twist
        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = angular_z
        self.twist_pub.publish(twist)

        self.step += 1

    def avoid_wall(self):
        # check how stacked for each direction
        front_stacked = torch.sum(self.state.lidar[0][0][:45] < DIST_TO_WALL_TH) \
                + torch.sum(self.state.lidar[0][0][315:] < DIST_TO_WALL_TH)
        left_stacked = torch.sum(self.state.lidar[0][0][45:135] < DIST_TO_WALL_TH)
        rear_stacked = torch.sum(self.state.lidar[0][0][135:225] < DIST_TO_WALL_TH)
        right_stacked = torch.sum(self.state.lidar[0][0][:315] < DIST_TO_WALL_TH)
        # if total of stacked is larger than threshold, recover stacked status
        if front_stacked + left_stacked + rear_stacked + right_stacked > NUM_LASER_CLOSE_TO_WALL_TH:
            print("### stacked ###")
            avoid = True
            SPEED = 0.4
            RAD = 3.14
            # decide where to go for recovering stacked status
            _, linear_x, angular_z = min([
                (front_stacked, SPEED, .0),
                (left_stacked, .0, RAD / 2), 
                (rear_stacked, -SPEED, .0),
                (right_stacked, .0, -RAD / 2),
            ], key=lambda e: e[0])
            
        else:
            avoid = False
            linear_x = None
            angular_z = None

        return avoid, linear_x, angular_z

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

    def init_amcl_pose(self):
        amcl_init_pub = rospy.Publisher('initialpose', PoseWithCovarianceStamped, queue_size=1)
        amcl_pose = PoseWithCovarianceStamped()
        amcl_pose.header.stamp = rospy.Time.now()
        amcl_pose.header.frame_id = "map"
        amcl_pose.pose.pose.position.x = -1.3
        amcl_pose.pose.pose.position.y = 0.0
        amcl_pose.pose.pose.position.z = 0.0
        amcl_pose.pose.pose.orientation.x = 0.0
        amcl_pose.pose.pose.orientation.y = 0.0
        amcl_pose.pose.pose.orientation.z = 0.0
        amcl_pose.pose.pose.orientation.w = 1.0
        amcl_init_pub.publish(amcl_pose)

    def stop(self):
        self.pause_service()

    def restart(self):
        self.episode += 1

        # restart judge server
        resp = send_to_judge(JUDGE_URL + "/warState/state", {"state": "running"})

        # restart gazebo physics
        self.unpause_service()

        # reset amcl pose
        self.init_amcl_pose()

        print("restart the game")

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
        subprocess.call('bash ../catkin_ws/src/burger_war_dev/burger_war_dev/scripts/reset.sh', shell=True)

        # reset robot's positions
        self.move_robot("red_bot", (0.0, -1.3, 0.0), (0, 0, 1.57))
        self.move_robot("blue_bot", (0.0, 1.3, 0.0), (0, 0, -1.57))

    def train(self, n_epochs=20):
        for epoch in range(n_epochs):
            print("episode {}: epoch {}".format(self.episode, epoch))
            self.agent.update_policy_network()
    
    def run(self, rospy_rate=1):

        r = rospy.Rate(rospy_rate)

        while not rospy.is_shutdown():
            
            while not all([v is not None for v in [self.lidar_ranges, self.my_pose, self.image]]):
                pass

            if self.game_state == "stop":
                
                if not self.debug:
                    break

                # stop the game
                self.stop()

                # offline learning
                if not self.online:
                    self.train(n_epochs=EPOCHS)

                # update target q function
                if self.episode % UPDATE_Q_FREQ == 0:
                    self.agent.update_target_network()

                # save model
                self.agent.save(self.save_path)

                # reset the game
                self.reset()

                time.sleep(1)

                # restart the game
                self.restart()

            elif self.game_state == "running":
                self.strategy()

                # online learning
                if self.online and self.debug:
                    self.agent.update_policy_network()

            r.sleep()

    
if __name__ == "__main__":

    rospy.init_node('dqn_run')
    JUDGE_URL = rospy.get_param('/send_id_to_judge/judge_url')

    try:
        ROBOT_NAME = rosparam.get_param('DQNRun/side')
    except:
        ROBOT_NAME = rosparam.get_param('enemyRun/side')

    # parameters

    ONLINE = False
    POLICY = "epsilon"
    DEBUG = True
    SAVE_PATH = "../catkin_ws/src/burger_war_dev/burger_war_dev/scripts/models/test.pth" 
    LOAD_PATH = None
    MANUAL_AVOID = True

    # wall avoidance
    DIST_TO_WALL_TH = 0.25  #[m]
    NUM_LASER_CLOSE_TO_WALL_TH = 90

    # action lists
    VEL = 0.4
    OMEGA = 1
    ACTION_LIST = [
        [0, 0],
        [VEL, 0],
        [-VEL, 0],
        [0, OMEGA],
        [0, -OMEGA],
    ]

    # agent config
    UPDATE_Q_FREQ = 5
    BATCH_SIZE = 32
    MEM_CAPACITY = 2000
    GAMMA = 0.99
    PRIOTIZED = True
    LR = 0.0005
    EPOCHS = 20

    # time freq [Hz]
    RATE = 1

    try:
        bot = DQNBot(robot=ROBOT_NAME, online=ONLINE, policy_mode=POLICY, debug=DEBUG, save_path=SAVE_PATH, load_path=LOAD_PATH, manual_avoid=MANUAL_AVOID)
        bot.run(rospy_rate=RATE)

    except rospy.ROSInterruptException:
        pass
