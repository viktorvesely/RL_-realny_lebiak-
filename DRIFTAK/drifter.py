import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow import keras
from typing import Any, List, Sequence, Tuple


from gaussianNoise import GaussianNoise


class Drifter(tf.keras.Model):

    tau = 0.1
    max_nosie_std = 1.5
    gamma = 0.98

    critic_lr = 0.004
    actor_lr = 0.002

    stable_noise = True
    stable_noise_std = 0.15

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

        self.noise = GaussianNoise(
            Drifter.stable_noise_std if Drifter.stable_noise else Drifter.max_nosie_std,
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

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_weights)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_weights)
        )

        with tf.GradientTape() as tape:
            actions_hat = self.actor(states, training=True)
            critic_value = self.critic([states, actions_hat], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_weights)
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
        actor.add(layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)))
        actor.add(layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)))
        actor.add(layers.Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)))
        actor.add(layers.Dense(
            self.num_actions(),
            activation='tanh',
            bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.008, seed=None)))

        return actor
        

    def learn(self, batch):

        states, actions, rewards, next_states = batch

        self.__at_least_one_training = True
    
        return self.update(
            tf.convert_to_tensor(states),
            tf.convert_to_tensor(actions),
            tf.convert_to_tensor(rewards),
            tf.convert_to_tensor(next_states)
        )


    def init_critic(self):

        state_input = keras.Input(shape=self.state_shape)

        state_output = layers.Dense(128,
            activation='relu',
        )(state_input)

        state_output = layers.Dense(64,
            activation='relu',
        )(state_output)

        state_output = layers.Dense(32,
            activation='relu',
        )(state_output)


        action_input = keras.Input(shape=(self.num_actions()))
        action_out = layers.Dense(128, activation="relu")(action_input)
        action_out = layers.Dense(64, activation="relu")(action_out)

        concat = layers.Concatenate()([state_output, action_out])

        out = layers.Dense(128, activation="relu")(concat)
        out = layers.Dense(64, activation="linear")(out)
        outputs = layers.Dense(1, activation="linear")(out)

        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def Normalizeto01(self, data):
        return (data + 1)/2
        

    def __call__(self, state, training=True):
        # Convert to batch
        state_batch = tf.expand_dims(tf.convert_to_tensor(state), 0)

        # Convert back to one sample
        actions = tf.squeeze(self.actor(state_batch)).numpy()

        noise = self.noise()

        boundaries = self.action_space

        actions = np.clip(actions, boundaries[0], boundaries[1])
        #actions[1] = self.Normalizeto01(actions[1])

        if not Drifter.stable_noise:
            noise *= 1 / self.t

        if training: 
            actions += noise
            actions = np.clip(actions, boundaries[0], boundaries[1])
            #actions[1] = self.Normalizeto01(actions[1])


        return actions

    def load_for_exploitation(self):
        self.actor.load_weights("./brains/weights")

    def save_for_exploitation(self):
        self.actor.save_weights("./brains/weights")