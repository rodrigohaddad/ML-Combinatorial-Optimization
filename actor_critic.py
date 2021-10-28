from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np


class Actor:
    def __init__(self, alpha, beta,
                 input_dims=8, layer1_size=1024,
                 layer2_size=512, n_actions=4,
                 gamma=0.99):
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        self.action_space = [i for i in range(n_actions)]

    def build_actor_critic_network(self):
        input = Input(shape=(self.input_dims,))
        self.delta = Input(shape=[1])
        dense1 = Dense(self.fc1_dims)
        dense2 = Dense(self.fc1_dims, activation='relu')(dense1)
        probs = Dense(1, activation='linear')(dense2)
        values = Dense(1, activation='linear')(dense2)

        actor = Model(input=[input, self.delta], output=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=self.custom_loss)

        critic = Model(input=[input], output=[values])
        critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')

        policy = Model(input=[input], output=[probs])

        return actor, critic, policy

    def custom_loss(self, y_true, y_pred):
        loss = 1e-8
        out = K.clip(y_pred, loss, 1-loss)
        log_lik = y_true*K.log(out)

        return K.sum(-log_lik*self.delta)

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        return np.random.choice(self.action_space, p=probabilities)

    def learn(self, state, state_, action, reward, done):
        state = state[np.newaxis, :]
        state_ = state[np.newaxis, :]

        critic_value_ = self.critic.predict(state_)
        critic_value = self.critic.predict(state)

        target = reward + self.gamma*critic_value_*(1-int(done))
        delta = target - critic_value

        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1.0

        self.actor.fit([state, delta], actions, verbose=0)
        self.critic.fit(state, target, verbose=0)


