

class EnvWrapper:
    def __init__(self, env):
        self.env = env

    def reset(self):
        self.episode_reward = 0
        self.episode_length = 0
        return self.env.reset()

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        self.episode_length += 1
        return next_obs, reward, done, info

    def get_episode_reward(self):
        return self.episode_reward

    def get_episode_length(self):
        return self.episode_length
