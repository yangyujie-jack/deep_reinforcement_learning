from easydict import EasyDict as edict
import gym


class Config:
    def __init__(self):
        # env
        env = edict()
        env.name = "CartPole-v0"

        # model
        model = edict()

        # dqn
        model.dqn = edict()
        model.dqn.epsilon = 0.1
        model.dqn.gamma = 0.99

        # learner
        learner = edict()
        learner.batch_size = 32

        # replay
        replay = edict()
        replay.capacity = int(1e4)

        self.env = env
        self.model = model
        self.learner = learner
        self.replay = replay

        self.setup()

    def setup(self):
        _env = gym.make(self.env.name)
        self.env.obs_shape = _env.observation_space.shape
        self.env.n_action = _env.action_space.n
        _env.close()


if __name__ == '__main__':
    cfg = Config()
    print(cfg.env.obs_shape, cfg.env.n_action)
