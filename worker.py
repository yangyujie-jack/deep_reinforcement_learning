import gym
from env_wrapper import EnvWrapper
from torch.utils.tensorboard import SummaryWriter


class Worker:
    def __init__(self, config, model, replay):
        self.config = config
        self.model = model
        self.replay = replay
        self.env = EnvWrapper(gym.make(config.env.name))
        self.writer = SummaryWriter(config.log.path)
        self.write_num = 0

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

    def end_episode(self):
        ep_r = self.env.get_episode_reward()
        ep_l = self.env.get_episode_length()
        print(f"[worker]episode reward: {ep_r}, episode length: {ep_l}")
        self.writer.add_scalar("episode reward", ep_r, self.write_num)
        self.writer.add_scalar("episode length", ep_l, self.write_num)
        self.write_num += 1

