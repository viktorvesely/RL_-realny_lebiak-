
import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from Params import Params

from networks import Actor, Critic
from noise import OrnsteinUhlenbeckActionNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Drifter:
    

    def __init__(self, action_space, state_shape):

        self.pars = Params()

        self.action_space = action_space
        self.state_shape = state_shape
        self.action_lows = torch.Tensor(action_space[0]).to(device)
        self.action_highs = torch.Tensor(action_space[1]).to(device)

        self.critic = Critic(self.state_shape, self.action_space).to(device)
        self.actor = Actor(self.state_shape, self.action_space).to(device)

        self.critic_target = Critic(self.state_shape, self.action_space).to(device)
        self.actor_target = Actor(self.state_shape, self.action_space).to(device)

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        self.ou_noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(self.num_actions()),
            sigma=self.pars.get("noise_std") * np.ones(self.num_actions())
        )

        self.actor_optimizer = Adam(
            self.actor.parameters(),
            lr=self.pars.get("actor_lr"),
            weight_decay=self.pars.get("actor_weight_decay")
        )

        self.critic_optimizer = Adam(
            self.critic.parameters(),
            lr=self.pars.get("critic_lr"),
            weight_decay=self.pars.get("critic_weight_decay")
        ) 


    def soft_update(self, target, source, tau):
        for wt, w in zip(target.parameters(), source.parameters()):
            wt.data.copy_(w.data * tau +  wt.data * (1.0 - tau))

    def hard_update(self, target, source):
        for wt, w in zip(target.parameters(), source.parameters()):
            wt.data.copy_(w.data)

    def sync_targets(self):
        self.soft_update(self.critic_target, self.critic, self.pars.get("tau"))
        self.soft_update(self.actor_target, self.actor, self.pars.get("tau"))

    def num_actions(self):
        return self.action_space.T.shape[0]

    def on_episode(self):
        self.ou_noise.reset()

    def __call__(self, state, training=True):

        x = torch.Tensor([state]).to(device)

        self.actor.eval() 
        action = self.actor(x)
        self.actor.train() 
        action = action.data

        if training:
            noise = torch.Tensor(self.ou_noise.noise()).to(device)
            action += noise

        action = action.clamp(self.action_lows, self.action_highs)

        return np.squeeze(action.cpu().numpy())
    
    def update(self, states, actions, rewards, next_states, dones):

        next_actions = self.actor_target(next_states)
        # TODO what exactly is detach
        Q_target_next = self.critic_target(next_states, next_actions.detach())

        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1)

        gamma = self.pars.get("gamma")
        td_target = rewards + (1 - dones) * gamma * Q_target_next

        # Critic update step
        self.critic_optimizer.zero_grad()
        Q_hat = self.critic(states, actions)
        value_loss = F.mse_loss(Q_hat, td_target.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(states, self.actor(states))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        return value_loss.item(), policy_loss.item()

        

    def learn(self, batch):

        states, actions, rewards, next_states, dones = batch
    
        return self.update(
            torch.Tensor(states).to(device),
            torch.Tensor(actions).to(device),
            torch.Tensor(rewards).to(device),
            torch.Tensor(next_states).to(device),
            torch.Tensor(dones).to(device)
        )

    def set_eval(self):
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()


    def save_model(self, path=None):
        if path is None:
            torch.save(self.actor, "./brains/torch_actor.pt")
        else:
            torch.save(self.actor, path)


    def load_model(self, path=None):
        if path is None:
            self.actor = torch.load("./brains/torch_actor.pt")
        else:
            self.actor = torch.load(path)

