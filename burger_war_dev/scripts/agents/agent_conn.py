#!python3
#-*- coding: utf-8 -*-

import sys
from .agent import Agent
from .connection import QueueCommunicator
from .connection import connect_socket_connection, accept_socket_connections

class AgentServer(QueueCommunicator):
    def __init__(self, port=5010):
        """
        Args:
            port (int): port to listen
        """
        super(AgentServer, self).__init__()
        self.shutdown_flag = False
        self.port = port

        # wait for connection
        param = self._wait()

        # create agent instance
        print('agent client found. instantiating agent...')
        self.agent = Agent(**param['args'])

    def _wait(self):
        print('waiting agent client: port %d' % self.port)
        conn_acceptor = accept_socket_connections(port=self.port, timeout=0.5)
        connected = False
        while not self.shutdown_flag:
            try:
                conn = next(conn_acceptor)
                if conn is not None:
                    param = conn.recv()
                    print('accepted connection from %s!' % param['conn']['address'])
                    self.add(conn)
                    connected = True
                    break
            except KeyboardInterrupt:
                print('interrupted')
                self.shutdown_flag = True
                break
        if not connected:
            print('failed to connect')
            raise Exception
        return param

    def run(self):
        """
        run agent according to request
        """
        while not self.shutdown_flag:
            try:
                # wait for connection
                if len(self.conns) == 0:
                    self._wait()

                # receive message
                conn, (command, args) = self.recv()

                # process according to request
                if command == 'update_policy_network':
                    self.agent.update_policy_network()
                elif command == 'get_action':
                    action = self.agent.get_action(**args)
                    self.send(conn, action)
                elif command == 'memorize':
                    self.agent.memorize(**args)
                elif command == 'save_model':
                    self.agent.save_model(**args)
                elif command == 'load_model':
                    self.agent.load_model(**args)
                elif command == 'save_memory':
                    self.agent.save_memory(**args)
                elif command == 'load_memory':
                    self.agent.load_memory(**args)
                elif command == 'update_target_network':
                    self.agent.update_target_network()
                elif command == 'detach':
                    self.disconnect(conn)
                elif command == 'shutdown':
                    self.shutdown_flag = True
                else:
                    print('unknown command: {}'.format(command))
            except KeyboardInterrupt:
                print('interrupted')
                self.shutdown_flag = True
                break

class AgentClient(QueueCommunicator):
    def __init__(self, server_address, port, num_actions, batch_size=32, capacity=10000, gamma=0.99, prioritized=True, lr=0.0005):
        """
        Args:
            num_actions (int): number of actions to output
            batch_size (int): batch size
            capacity (int): capacity of memory
            gamma (int): discount rate
        """
        super(AgentClient, self).__init__()

        # pack arguments
        param = {
            'conn': {
                'address': server_address
            },
            'args': {
                'num_actions': num_actions,
                'batch_size': batch_size,
                'capacity': capacity,
                'gamma': gamma,
                'prioritized': prioritized,
                'lr': lr
            }
        }
        
        # connect to server and wait for first response
        self.server_conn = connect_socket_connection(server_address, port)
        self.server_conn.send(param)
        self.add(self.server_conn)

    def update_policy_network(self):
        """
        update policy network model
        Args:
            
        """
        self.send(self.server_conn, (sys._getframe().f_code.co_name, {}))

    def get_action(self, state, episode, policy_mode, debug):
        """
        get action
        Args:
            state (State): state including lidar, map and image
            episode (int): episode
        Return:
            action (Tensor): action (number)
        """
        args = {
            'state': state,
            'episode': episode,
            'policy_mode': policy_mode,
            'debug': debug
        }
        self.send(self.server_conn, (sys._getframe().f_code.co_name, args))
        self.server_conn, action = self.recv()
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
        args = {
            'state': state,
            'action': action,
            'state_next': state_next,
            'reward': reward
        }
        self.send(self.server_conn, (sys._getframe().f_code.co_name, args))

    def save_model(self, path):
        """
        save model
        Args:
            path (str): path to save
        """
        self.send(self.server_conn, (sys._getframe().f_code.co_name, { 'path': path }))

    def load_model(self, path):
        """
        load model
        Args:
            path (str): path to load
        """
        self.send(self.server_conn, (sys._getframe().f_code.co_name, { 'path': path }))

    def save_memory(self, path):
        """
        save memory
        Args:
            path (str): path to save
        """
        self.send(self.server_conn, (sys._getframe().f_code.co_name, { 'path': path }))

    def load_memory(self, path):
        """
        load memory
        Args:
            path (str): path to load
        """
        self.send(self.server_conn, (sys._getframe().f_code.co_name, { 'path': path }))
    def update_target_network(self):
        """
        update target network model
        """
        self.send(self.server_conn, (sys._getframe().f_code.co_name, {}))

    def detach(self):
        """
        detach agent
        """
        self.send(self.server_conn, (sys._getframe().f_code.co_name, {}))