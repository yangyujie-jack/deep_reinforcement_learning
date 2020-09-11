from easydict import EasyDict as edict
import gym
import os
import time


class Config:
    def __init__(self):
        # env
        env = edict()
        env.name = "CartPole-v0"

        # model
        model = edict()

        # dqn
        model.dqn = edict()
        model.dqn.epsilon_start = 1.0
        model.dqn.epsilon_final = 0.01
        model.dqn.epsilon_decay = 100
        model.dqn.gamma = 0.99

        # learner
        learner = edict()
        learner.batch_size = 32

        # optimizer
        optimizer = edict()
        optimizer.lr = 1e-3
        optimizer.momentum = 0.95
        optimizer.eps = 0.01

        # replay
        replay = edict()
        replay.capacity = int(1e4)

        # log
        log = edict()
        log.path = os.path.abspath(os.path.dirname(__file__)) + '/log/'

        self.env = env
        self.model = model
        self.learner = learner
        self.optimizer = optimizer
        self.replay = replay
        self.log = log

        self.setup()

    def setup(self):
        _env = gym.make(self.env.name)
        self.env.obs_shape = _env.observation_space.shape
        self.env.n_action = _env.action_space.n
        _env.close()

        self.log.path += time.strftime("%Y-%m-%d_%H.%M", time.localtime())


if __name__ == '__main__':
    cfg = Config()
    print(cfg.env.obs_shape, cfg.env.n_action)
