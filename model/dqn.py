import torch
import torch.nn as nn
import random
import numpy as np


class DQN(nn.Module):
    def __init__(self, config):
        super(DQN, self).__init__()
        self.config = config
        obs_shape = config.env.obs_shape
        n_action = config.env.n_action
        self.layers = nn.Sequential(
            nn.Linear(obs_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_action)
        )

    def forward(self, x):
        return self.layers(x)

    def choose_action(self, obs):
        """
        epsilon-greedy
        """
        if random.random() > self.config.model.dqn.epsilon:
            obs = torch.tensor(obs).float()
            q = self.forward(obs)
            action = torch.argmax(q).numpy()
        else:
            action = random.randrange(self.config.env.n_action)
        return action

    def cal_loss(self, batch):
        obs = torch.tensor(np.array(batch[0])).float()
        action = torch.tensor(np.array(batch[1])).long()
        reward = torch.tensor(np.array(batch[2])).float()
        next_obs = torch.tensor(np.array(batch[3])).float()
        done = torch.tensor(np.array(batch[4])).float()

        qs = self.forward(obs)
        next_qs = self.forward(next_obs).detach()

        q = torch.gather(qs, 1, action.unsqueeze(1)).squeeze()
        next_q = torch.max(next_qs, 1)[0]
        td_target = reward + self.config.model.dqn.gamma * (1 - done) * next_q

        mseloss = nn.MSELoss()
        loss = mseloss(td_target, q)
        return loss


if __name__ == '__main__':
    from config import Config

    cfg = Config()
    model = DQN(cfg)

    obs = torch.rand(4).float()
    action = model.choose_action(obs)
    print(action)