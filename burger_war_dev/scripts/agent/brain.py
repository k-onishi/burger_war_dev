import random

import numpy as np
import pickle
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable

from network.masknet import MaskNet
from state import State
from transition import Transition
from random_memory import RandomMemory


class Brain():
    def __init__(self, batch_size, capacity, gamma, learning_rate, num_actions,
            duel=True):
        self.batch_size = batch_size
        self.gamma = gamma
        self.num_actions = num_actions
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.memory = RandomMemory(capacity)

        self.policy_net = MaskNet(num_actions, duel=duel).to(self.device)
        self.target_net = MaskNet(num_actions, duel=duel).to(self.device)

        print("using device: {}".format(self.device))

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def replay(self):
        if len(self.memory) < self.batch_size:
            print("too few memories")
            return

        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))
        batch_state = State(*zip(*batch.state))
        batch_next_state = State(*zip(*batch.next_state))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(self.device)

        pose_batch = Variable(torch.cat(batch_state.pose)).to(self.device)
        lidar_batch = Variable(torch.cat(batch_state.lidar)).to(self.device)
        image_batch = Variable(torch.cat(batch_state.image)).to(self.device)
        mask_batch = Variable(torch.cat(batch_state.mask)).to(self.device)

        action_batch = Variable(torch.cat(batch.action)).to(self.device)
        reward_batch = Variable(torch.cat(batch.reward)).to(self.device)

        non_final_next_poses = Variable(torch.cat([s for s in batch_next_state.pose if s is not None])).to(self.device)
        non_final_next_lidars = Variable(torch.cat([s for s in batch_next_state.lidar if s is not None])).to(self.device)
        non_final_next_images = Variable(torch.cat([s for s in batch_next_state.image if s is not None])).to(self.device)
        non_final_next_masks = Variable(torch.cat([s for s in batch_next_state.mask if s is not None])).to(self.device)

        self.policy_net.eval()

        state_action_values = self.policy_net(pose_batch, lidar_batch, image_batch, mask_batch).gather(1, action_batch)

        next_state_values = Variable(torch.zeros(self.batch_size).type(torch.FloatTensor)).to(self.device)

        a_m = Variable(torch.zeros(self.batch_size).type(torch.LongTensor)).to(self.device)
        a_m[non_final_mask] = self.policy_net(non_final_next_poses,
                                            non_final_next_lidars,
                                            non_final_next_images,
                                            non_final_next_masks).max(1)[1].detach()

        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        next_state_values[non_final_mask] = self.target_net(
                                                non_final_next_poses,
                                                non_final_next_lidars,
                                                non_final_next_images,
                                                non_final_next_masks
                                            ).gather(1, a_m_non_final_next_states).detach().squeeze()

        expected_state_action_values = reward_batch + self.gamma * next_state_values
        expected_state_action_values = expected_state_action_values.unsqueeze(1)

        self.policy_net.train()  # TODO: No need?

        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print("Loss: {}".format(loss.item()))

    def decide_action(self, state, episode, is_training=False):
        self.policy_net.eval()

        if episode < 50:
            epsilon = 0.25
        elif episode < 100:
            epsilon = 0.15
        else:
            epsilon = 0.05

        if is_training and epsilon <= np.random.uniform(0, 1):
            print("Random Action")
            action = torch.LongTensor([[random.randrange(self.num_actions)]]).to(self.device)
        else:
            input_pose = Variable(state.pose).to(self.device)
            input_lidar = Variable(state.lidar).to(self.device)
            input_image = Variable(state.image).to(self.device)
            input_mask = Variable(state.mask).to(self.device)

            output = self.policy_net(input_pose, input_lidar, input_image, input_mask)
            action = output.data.max(1)[1].view(1, 1)

        return action

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.update_target_network()

    def save_memory(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.memory, f)

    def load_memory(self, path):
        with open(path, 'rb') as f:
            self.memory = pickle.load(f)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
