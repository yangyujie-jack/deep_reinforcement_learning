import torch.optim as optim


class Learner:
    def __init__(self, config, model, replay):
        self.config = config
        self.model = model
        self.replay = replay
        self.optimizer = optim.RMSprop(model.parameters(),
                                       lr=config.optimizer.lr,
                                       momentum=config.optimizer.momentum,
                                       eps=config.optimizer.eps)

    def learn(self):
        batch = self.replay.sample(self.config.learner.batch_size)
        if not batch:
            return
        loss = self.model.cal_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
