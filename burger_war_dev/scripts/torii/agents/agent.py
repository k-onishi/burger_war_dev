﻿#!python3
#-*- coding: utf-8 -*-

from brain import Brain


class Agent:
    """
    An agent for DDQN
    """
    def __init__(self, num_actions, batch_size=32, capacity=10000, gamma=0.99, prioritized=True, lr=0.0005):
        """
        Args:
            num_actions (int): number of actions to output
            batch_size (int): batch size
            capacity (int): capacity of memory
            gamma (int): discount rate
        """
        self.brain = Brain(num_actions, batch_size, capacity, gamma, prioritized, lr)  # エージェントが行動を決定するための頭脳を生成

    def update_policy_network(self):
        """
        update policy network model
        Args:
            
        """
        self.brain.replay()

    def get_action(self, state, episode, policy_mode, debug):
        """
        get action
        Args:
            state (State): state including lidar, map and image
            episode (int): episode
        Return:
            action (Tensor): action (number)
        """
        action = self.brain.decide_action(state, episode, policy_mode, debug)
        return action

    def memorize(self, state, action, state_next, reward):
        """
        memorize current state, action, next state and reward
        Args:
            state (dict): current state
            action (Tensor): action
            state_next (dict): next state
            reward (int): reward
        """
        self.brain.memory.push(state, action, state_next, reward)

    def save_model(self, path):
        """
        save model
        Args:
            path (str): path to save
        """
        self.brain.save_model(path)

    def load_model(self, path):
        """
        load model
        Args:
            path (str): path to load
        """
        self.brain.load_model(path)

    def save_memory(self, path):
        """
        save memory
        Args:
            path (str): path to save
        """
        self.brain.save_memory(path)

    def load_memory(self, path):
        """
        load memory
        Args:
            path (str): path to load
        """
        self.brain.load_memory(path)

    def update_target_network(self):
        """
        update target network model
        """
        self.brain.update_target_network()
    
    def detach(self):
        """
        detach agent (for server-client implementation)
        """
        pass
