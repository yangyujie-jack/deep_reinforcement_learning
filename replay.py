from collections import deque
import random


class Replay:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, sample):
        """
        tuple, (obs, action, reward, next_obs, done)
        """
        self.buffer.append(sample)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return False
        obs, action, reward, next_obs, done = zip(*random.sample(self.buffer, batch_size))
        return obs, action, reward, next_obs, done

