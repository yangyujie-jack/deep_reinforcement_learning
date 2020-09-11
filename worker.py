import gym
from env_wrapper import EnvWrapper


class Worker:
    def __init__(self, config, model, replay):
        self.config = config
        self.model = model
        self.replay = replay
        self.env = EnvWrapper(gym.make(config.env.name))

    def choose_action(self, obs):
        return self.model.choose_action(obs)

    def store_sample(self, obs, action, reward, next_obs, done):
        self.replay.push((obs, action, reward, next_obs, done))

    def begin_episode(self):
        self.obs = self.env.reset()
        self.done = False

    def run_one_step(self):
        action = self.choose_action(self.obs)
        next_obs, reward, done, _ = self.env.step(action)
        self.store_sample(self.obs, action, reward, next_obs, done)
        self.obs = next_obs
        self.done = done
        if done:
            print(f"[worker]episode reward: {self.env.get_episode_reward()}, "
                  f"episode length: {self.env.get_episode_length()}")

