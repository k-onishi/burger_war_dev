#!/usr/bin/env python

from argparse import ArgumentParser
from copy import deepcopy
import json
import os
import requests

import cv2
from cv_bridge import CvBridge, CvBridgeError
import rospy
import rosparam
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from std_srvs.srv import Empty
import torch
from torchvision.transforms import ToTensor

from agent.agent import Agent
from agent.state import State


JUDGE_URL = ""
FIELD_MARKERS = [
    "Tomato_N", "Tomato_S", "Omelette_N", "Omelette_S", "Pudding_N", "Pudding_S",
    "OctopusWiener_N", "OctopusWiener_S", "FriedShrimp_N", "FriedShrimp_E", "FriedShrimp_W", "FriedShrimp_S"
]
ROBOT_MARKERS = {
    "r": ["RE_B", "RE_L", "RE_R"],
    "b": ["BL_B", "BL_L", "BL_R"]
}


class CallbackInputs():
    def __init__(self):
        self.reset()

    def reset(self):
        self.lidar_ranges = None
        self.my_pose = None
        self.image = None
        self.mask = None

    @property
    def is_ready(self):
        return all([
            v is not None for v in [
                self.lidar_ranges,
                self.my_pose,
                self.image,
                self.mask,
            ]
        ])

    @property
    def state(self):
        return State(
            self.my_pose,
            self.lidar_ranges,
            self.image,
            self.mask,
        )

class DQNBot():
    def __init__(self, batch_size, capacity, episode, gamma, learning_rate, action_list,
            model_path=None, memory_path=None, is_training=True):
        self.robot = "r"
        self.enemy = "b"
        self.action_list = action_list
        self.callback_inputs = CallbackInputs()
        self.marker_list = ROBOT_MARKERS[self.robot] \
                + ROBOT_MARKERS[self.enemy] \
                + FIELD_MARKERS
        self.score = {marker: 0 for marker in self.marker_list}
        self.past_score = {marker: 0 for marker in self.marker_list}
        self.action = None
        self.state = None
        self.episode = episode
        self.is_training = False
        self.model_path = model_path
        self.memory_path = memory_path
        self.is_training = is_training
        self.game_state = None

        self.lidar_sub = rospy.Subscriber('scan', LaserScan, self.callback_lidar)
        self.image_sub = rospy.Subscriber('image_raw', Image, self.callback_image)
        self.amcl_sub = rospy.Subscriber("amcl_pose", PoseWithCovarianceStamped, self.callback_amcl)
        self.state_sub = rospy.Timer(rospy.Duration(0.5), self.callback_warstate)

        self.twist_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        self.state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.pause_service = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_service = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

        self.agent = Agent(
            batch_size,
            capacity,
            gamma,
            learning_rate,
            len(self.action_list)
        )

        if self.model_path is not None and os.path.exists(self.model_path):
            print("Load model from {}".format(self.model_path))
            self.agent.load_model(self.model_path)
        if self.memory_path is not None and os.path.exists(self.memory_path):
            print("Load memory from {}".format(self.memory_path))
            self.agent.load_memory(self.memory_path)

    def callback_lidar(self, lidar):
        lidar_data = torch.FloatTensor(lidar.ranges).view(1, 1, -1)
        lidar_data[lidar_data>3.5] = 3.5
        self.callback_inputs.lidar_ranges = lidar_data / 3.5

    def callback_image(self, image):
        try:
            bridge = CvBridge()
            img = bridge.imgmsg_to_cv2(image, "bgr8")
            img = cv2.resize(img, (160, 95))
            deriv_x = ToTensor()(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5))
            deriv_y = ToTensor()(cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5))
            grad = torch.sqrt(deriv_x * deriv_x + deriv_y * deriv_y)
            self.callback_inputs.image = grad.unsqueeze(0)
            # print("deriv x: {}".format(deriv_x.size()))
            # print("deriv y: {}".format(deriv_y.size()))
            # print("image size: {}".format(self.callback_inputs.image.size()))
        except CvBridgeError as e:
            rospy.logerr(e)

    def callback_amcl(self, amcl):
        x = amcl.pose.pose.position.x
        y = amcl.pose.pose.position.y
        self.callback_inputs.my_pose = torch.FloatTensor([-y, x]).view(1, 2)

    def callback_warstate(self, event):
        json_dict = json.loads(requests.get(JUDGE_URL + "/warState").text)
        self.game_state = json_dict['state']

        if self.game_state == "running":
            for tg in json_dict["targets"]:
                if tg["player"] == self.robot:
                    self.score[tg["name"]] = int(tg["point"])
                elif tg["player"] == self.enemy:
                    self.score[tg["name"]] = -int(tg["point"])

            markers_mask = []
            for marker in self.marker_list:
                if self.score[marker] > 0:
                    markers_mask.append(0)
                elif self.score[marker] == 0:
                    markers_mask.append(1)
                else:
                    markers_mask.append(2)
            self.callback_inputs.mask = torch.FloatTensor(markers_mask).view(1, 18)

    def get_reward(self, past_score, current_score):
        diff_score = {tag: current_score[tag] - past_score[tag] for tag in self.score.keys()}
        score = 0
        reward = .0
        for tag in self.score.keys():
            if past_score[tag] <= 0 and current_score[tag] > 0:
                reward += 0.3
        for v in diff_score.values():
            score += v
        if score > 0:
            reward += 0.7
        elif score == 0:
            reward += 0.3
        return reward

    def strategy(self):
        self.past_state = self.state
        self.state = self.callback_inputs.state

        if self.action is not None:
            current_score = deepcopy(self.score)
            reward = self.get_reward(self.past_score, current_score)
            print("reward: {}".format(reward))
            self.past_score = current_score
            reward = torch.LongTensor([reward])
            self.agent.memorize(self.past_state, self.action, self.state, reward)

        self.action = self.agent.get_action(self.state, self.episode, self.is_training)
        choice = int(self.action.item())
        linear_x = self.action_list[choice][0]
        angular_z = self.action_list[choice][0]

        print("vel: {}, omega: {}".format(linear_x, angular_z))
        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = angular_z
        self.twist_pub.publish(twist)

    def train(self, num_epochs=20):
        for epoch in range(num_epochs):
            print("episode: {}: epoch: {}".format(self.episode, epoch))
            self.agent.update_policy_network()

    def run(self, rospy_rate=1):
        r = rospy.Rate(rospy_rate)

        print("start running")
        while not rospy.is_shutdown():
            if self.game_state == "running":
                if self.callback_inputs.is_ready:
                    self.strategy()
            if self.game_state == "stop":
                if self.is_training:
                    self.train()
                    if self.model_path is not None:
                        self.agent.save_model(self.model_path)
                    if self.memory_path is not None:
                        self.agent.save_memory(self.memory_path)
                break
            r.sleep()

        print("end running")


if __name__ == "__main__":
    batch_size = rosparam.get_param('DQNRun/batch_size')
    capacity = rosparam.get_param('DQNRun/capacity')
    episode = rosparam.get_param('DQNRun/episode')
    gamma = rosparam.get_param('DQNRun/gamma')
    learning_rate = rosparam.get_param('DQNRun/learning_rate')
    model_path = rosparam.get_param('DQNRun/model_path')
    memory_path = rosparam.get_param('DQNRun/memory_path')
    print("Got params")
    print("\tbatch_size: {}(type: {})".format(batch_size, type(batch_size)))
    print("\tcapacity: {}(type: {})".format(capacity, type(capacity)))
    print("\tepisode: {}(type: {})".format(episode, type(episode)))
    print("\tgamma: {}(type: {})".format(gamma, type(gamma)))
    print("\tlearning_rate: {}(type: {})".format(learning_rate, type(learning_rate)))

    rospy.init_node('dqn_run')

    # action lists
    VEL = 0.2
    OMEGA = 30 * 3.14/180
    ACTION_LIST = [
        [VEL, 0],
        [-VEL, 0],
        [0, 0],
        [0, OMEGA],
        [0, -OMEGA],
    ]
    JUDGE_URL = rospy.get_param('/send_id_to_judge/judge_url')

    try:
        bot = DQNBot(
            batch_size,
            capacity,
            episode,
            gamma,
            learning_rate,
            ACTION_LIST,
            model_path,
            memory_path,
            is_training=episode>=0,
        )
        bot.run()
    except rospy.ROSInterruptException:
        pass
