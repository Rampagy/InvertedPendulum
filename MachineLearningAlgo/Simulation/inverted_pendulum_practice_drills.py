import sys
sys.path.append("../Shared")

import train_model as tm
import NeuralNetwork as nn
import gym_custominvertedpendulum as gym

count = 0;
inverted_pend_eval_score = 0

#create env to initialize model
env = gym.make('CustomInvertedPendulum-v0')
obs_len = len(env.observation_space.low)
act_len = env.action_space.n

# create model
model = nn.Control_Model(obs_len, act_len)

# Phase 1
# train until the inverted pendulum gets a score above 100
# toggle between inverted pendulum and cartpole with increasing trained_threshold
# so that the model learns to flip the pole up and balance it once flipped up.

while (inverted_pend_eval_score <= 100):

    # on even numbers do cartpole
    if count%2==0:
        # practive balancing at the top
        env_name = 'CustomCartPole-v0'
        trained_threshold = 100
    # on odd numbers do inverted pendulum
    else:
        # practive flipping the pole up
        env_name = 'CustomInvertedPendulum-v0'
        # slowly demand that the inverted pendulum hold it longer and longer
        trained_threshold = 5 + int(count/2) * 5
        if trained_threshold > 100:
            trainted_threshold = 100

    env = gym.make(env_name)

    train_count, eval_score = tm.train_model(env=env,
        trained_threshold=trained_threshold, model=model)

    if env_name == 'CustomInvertedPendulum-v0':
        inverted_pend_eval_score = eval_score

    count += 1

# Phase 2
# Achieve a score average greater than 500 for 25 episodes

env_name = 'CustomInvertedPendulum-v0'
env = gym.make(env_name)

train_count, eval_score = tm.train_model(env=env, eval_episodes=25,
    trained_threshold=500, model=model)

# Phase 3
# Train until the model reaches a score of 1500 while random external forces
# are being applied to the system
# Ref: Disturbance Rejection

env_name = 'CustomInvertedPendulum_DisturbReject-v0'
env = gym.make(env_name)

train_count, eval_score = tm.train_model(env=env, eval_episodes=25,
    trained_threshold=700, model=model, train_episodes=50)
