import numpy as np

loss_every = 50

class Detective:

    def __init__(self, model, training):
        self.model = model
        self.training = training
        
        self.loss = []
        self.train_n = 0


    def on_train(self, actor_loss, critic_loss):
        if self.train_n % loss_every == 0:
            self.loss.append([actor_loss, critic_loss])
        self.train_n += 1
        
        ws = self.model

    def on_sync(self):
        pass

    def on_episode(self):
        pass

    def on_tick(self, state, action, reward, next_state):
        pass

    def on_end(self):
        if self.training:
            loss = np.array(self.loss).T
            np.save(self.path("loss"), loss)

    def path(self, name):
        return "./guts/" + name