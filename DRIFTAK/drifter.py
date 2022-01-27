import numpy as np
import tensorflow as tf
from random import random
from tensorflow.keras import layers, models
from tensorflow import keras
from typing import Any, List, Sequence, Tuple
import math


from gaussianNoise import GaussianNoise


class Drifter(tf.keras.Model):

    tau = 0.0001
    max_nosie_std = 1.5
    gamma = 0.98

    critic_lr = 0.004
    actor_lr = 0.002

    epsilon_noise = True
    epsilon = 1
    epsilon_decay = 0.0006
    epsilon_min = 0.08
    epsilon_noise_std = 0.9

    def __init__(
      self, 
      action_space: Tuple,
      state_shape: Tuple
      ):
        """Initialize."""
        super().__init__()

        self.action_space = action_space
        self.state_shape = state_shape
        self.actor = self.init_actor()
        self.critic = self.init_critic()

        self.actor_target = self.init_actor()
        self.critic_target = self.init_critic()

        self.sync_actor()
        self.sync_critic()

        self.critic_optimizer = tf.keras.optimizers.Adam(Drifter.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(Drifter.actor_lr)

        self.actor.compile(optimizer=self.actor_optimizer, loss="mse")
        self.critic.compile(optimizer=self.critic_optimizer, loss="mse")

        self.noise = GaussianNoise(
            Drifter.epsilon_noise_std if Drifter.epsilon_noise else Drifter.max_nosie_std,
            self.num_actions()
        )
        self.t = 0
        
        self.__at_least_one_training = False 

    @tf.function
    def update(
        self, states, actions, rewards, next_states,
    ):
        
        with tf.GradientTape() as tape:
            target_actions = self.actor_target(next_states, training=True)
            
            critic_value = rewards + Drifter.gamma * self.critic_target(
                [next_states, target_actions], training=True
            )
            
            critic_value_hat = self.critic([states, actions], training=True)
            
            critic_loss = tf.math.reduce_mean(
                tf.math.square(critic_value - critic_value_hat)
            )

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions_hat = self.actor(states, training=True)
            critic_value = self.critic([states, actions_hat])
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

        return tf.math.reduce_mean(actor_loss), tf.math.reduce_mean(critic_loss)


    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (wt, w) in zip(target_weights, weights):
            wt.assign(w * tau + wt * (1 - tau))


    def num_actions(self):
        return self.action_space.T.shape[0]

    def sync_actor(self):
        self.update_target(self.actor_target.variables, self.actor.variables, Drifter.tau)

    def sync_critic(self):
        self.update_target(self.critic_target.variables, self.critic.variables, Drifter.tau)
        
    def sync_targets(self):
        self.sync_critic()
        self.sync_actor()

    def init_actor(self):
        actor = models.Sequential()
        actor.add(layers.InputLayer(input_shape=self.state_shape))
        actor.add(layers.Dense(128, activation='relu'))
        actor.add(layers.Dense(64, activation='relu'))
        actor.add(layers.Dense(32, activation='relu'))
        actor.add(layers.Dense(
            self.num_actions(),
            activation='tanh'))

        return actor
        

    def learn(self, batch):

        states, actions, rewards, next_states = batch
        self.t += 1

        self.__at_least_one_training = True
    
        return self.update(
            tf.convert_to_tensor(states),
            tf.convert_to_tensor(actions),
            tf.convert_to_tensor(rewards),
            tf.convert_to_tensor(next_states)
        )


    def init_critic(self):

        state_input = keras.Input(shape=self.state_shape)

        state_output = layers.Dense(64,
            activation='relu',
            #kernel_initializer=tf.keras.initializers.Zeros()
        )(state_input)

        state_output = layers.Dense(32,
            activation='relu',
            #kernel_initializer=tf.keras.initializers.Zeros()
        )(state_output)


        action_input = keras.Input(shape=(self.num_actions()))
        action_out = layers.Dense(64, 
            activation="relu",
            #kernel_initializer=tf.keras.initializers.Zeros()
        )(action_input)
        action_out = layers.Dense(
            32,
            activation="relu",
            #kernel_initializer=tf.keras.initializers.Zeros()
        )(action_out)

        concat = layers.Concatenate()([state_output, action_out])

        out = layers.Dense(128,
         activation="relu",
        # kernel_initializer=tf.keras.initializers.Zeros()
        )(concat)
        out = layers.Dense(64,
         activation="relu", 
         #kernel_initializer=tf.keras.initializers.Zeros()
        )(out)
        outputs = layers.Dense(1,
         activation="linear",
         #kernel_initializer=tf.keras.initializers.Zeros()
         )(out)

        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def Normalizeto01(self, data):
        return (data + 1)/2
        
    def get_epsilon(self):
        return max(Drifter.epsilon_min, Drifter.epsilon * math.exp(-self.t * Drifter.epsilon_decay))

    def __call__(self, state, training=True):
        # Convert to batch
        state_batch = tf.expand_dims(tf.convert_to_tensor(state), 0)

        # Convert back to one sample
        actions = tf.squeeze(self.actor(state_batch)).numpy()

        noise = self.noise()

        boundaries = self.action_space

        actions = np.clip(actions, boundaries[0], boundaries[1])
        #actions[1] = self.Normalizeto01(actions[1])

        if not Drifter.epsilon_noise:
            noise *= 1 / self.t

        epsilon = self.get_epsilon()
        if training and (random() <= epsilon): 
            actions = noise
            actions = np.clip(actions, boundaries[0], boundaries[1])
            #actions[1] = self.Normalizeto01(actions[1])


        return actions

    def load_for_exploitation(self):
        self.actor.load_weights("./brains/weights")

    def save_for_exploitation(self):
        self.actor.save_weights("./brains/weights")