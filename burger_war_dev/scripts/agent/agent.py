from brain import Brain


class Agent():
    def __init__(self, batch_size, capacity, gamma, learning_rate, num_actions):
        self.brain = Brain(batch_size, capacity, gamma, learning_rate, num_actions)

    def update_policy_network(self):
        self.brain.replay()

    def get_action(self, state, episode, is_training=False):
        action = self.brain.decide_action(state, episode, is_training)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def save_model(self, path):
        self.brain.save_model(path)

    def load_model(self, path):
        self.brain.load_model(path)

    def save_memory(self, path):
        self.brain.save_memory(path)

    def load_memory(self, path):
        self.brain.load_memory(path)

    def update_target_network(self):
        self.brain.update_target_network()
