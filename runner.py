from replay import Replay
from worker import Worker
from model.dqn import DQN
from learner import Learner


class Runner:
    def __init__(self, config):
        self.config = config
        self.replay = Replay(config.replay.capacity)
        self.model = DQN(config)
        self.worker = Worker(config, self.model, self.replay)
        self.learner = Learner(config, self.model, self.replay)

    def run(self):
        while True:
            self.worker.begin_episode()
            while not self.worker.done:
                self.worker.run_one_step()
                self.learner.learn()


