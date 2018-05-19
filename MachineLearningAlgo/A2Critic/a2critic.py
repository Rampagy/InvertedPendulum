import sys
import gym_custominvertedpendulum as gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

EPISODES = 1000


# A2C(Advantage Actor-Critic) agent for the Cartpole
class A2CAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        self.test = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        # create model for policy network
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        self.batch_size = 64
        self.train_start = 1000
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        if self.load_model:
            self.actor.load_weights("./invpend_actor.h5")
            self.critic.load_weights("./invpend_critic.h5")

    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(100, input_dim=self.state_size, activation='relu',
                        kernel_initializer='glorot_uniform'))
        actor.add(Dense(100, activation='relu', kernel_initializer='glorot_uniform'))
        actor.add(Dense(100, activation='relu', kernel_initializer='glorot_uniform'))
        actor.add(Dense(self.action_size, activation='softmax',
                        kernel_initializer='glorot_uniform'))
        actor.summary()
        # See note regarding crossentropy in cartpole_reinforce.py
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(100, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear',
                         kernel_initializer='he_uniform'))
        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # update policy network every episode
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.value_size))
        advantages = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            value = self.critic.predict(state)[0]
            next_value = self.critic.predict(next_state)[0]

            states[i] = state
            if done:
                advantages[i][action] = reward - value
                targets[i][0] = reward
            else:
                advantages[i][action] = reward + self.discount_factor * (next_value) - value
                targets[i][0] = reward + self.discount_factor * next_value

        self.actor.fit(states, advantages, epochs=1, verbose=0)
        self.critic.fit(states, targets, epochs=1, verbose=0)


if __name__ == "__main__":
    # In case of CartPole-v1, maximum length of episode is 500
    env = gym.make('DampingPendulum-v0')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # make A2C agent
    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        time = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then give penalty of -100
            reward = reward if not done or time == 749 else -100

            if not agent.test:
                agent.replay_memory(state, action, reward, next_state, done)
                agent.train_model()

            time += 1
            score += reward
            state = next_state

            if done:
                # every episode, plot the play time
                score = score if time == 750.0 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./invpend_a2c.png")
                print("episode: {:3}   score: {:8.6}   memory length: {:4}"
                            .format(e, score, len(agent.memory)))

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(10, len(scores)):]) > 32000:
                    if not agent.test:
                        agent.actor.save_weights("./invpend_actor.h5")
                        agent.critic.save_weights("./invpend_critic.h5")
                    sys.exit()

        # save the model
        if e % 1 == 0 and not agent.test:
            agent.actor.save_weights("./invpend_actor.h5")
            agent.critic.save_weights("./invpend_critic.h5")
