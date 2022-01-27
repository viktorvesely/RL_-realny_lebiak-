import numpy as np

loss_every = 2000
weights_every = 400

debug_weights = False

class Detective:

    def __init__(self, model, training):
        self.model = model
        self.training = training
        
        self.loss = []
        self.train_n = 0

        self.aws = []
        self.cws = []
        self.taws = []
        self.tcws = []

        self.rewards = []

        self.cummulativeReward = 0


    def on_train(self, actor_loss, critic_loss):
        if self.train_n % loss_every == 0:
            self.loss.append([actor_loss, critic_loss])
        
        # if debug_weights: 

        #     if self.train_n % weights_every == 0:
        #         aws = self.model.actor.get_weights()
        #         cws = self.model.critic.get_weights()
                
        #         aw = [aws[0][1][4], aws[0][1][93], aws[4][42][15], aws[4][8][30], aws[6][2][0], aws[6][16][0]]
        #         cw = [cws[2][0][30], cws[2][1][3], cws[4][0][18], cws[4][0][5], cws[10][56][10], cws[10][78][32]]

        #         self.aws.append(aw)
        #         self.cws.append(cw)

        self.train_n += 1


    def on_episode(self):
        self.rewards.append(self.cummulativeReward)
        self.cummulativeReward = 0

    def on_tick(self, state, action, reward, next_state):
        if self.training:
            self.cummulativeReward += reward

    def on_end(self):
        if self.training:
            loss = np.array(self.loss).T
            np.save(self.path("loss"), loss)
            
            np.save(self.path("reward"), np.array(self.rewards))
            
            if debug_weights:
                np.save(self.path("aws"), np.array(self.aws))
                np.save(self.path("cws"), np.array(self.cws))
                np.save(self.path("taws"), np.array(self.taws))
                np.save(self.path("tcws"), np.array(self.tcws))

    def path(self, name):
        return "./guts/" + name