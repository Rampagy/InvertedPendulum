import sys
sys.path.append("../Shared")

import train_model as tm
import NeuralNetwork as nn
import gym_custominvertedpendulum as gym

count = 0;
inverted_pend_eval_score = 0

#create env to initialize model
env = gym.make('CustomInvertedPendulum-v0')

# create model
model = nn.Control_Model(len(env.observation_space.low))

train_count, eval_score = tm.train_model(env=env, eval_episodes=25,
    trained_threshold=400, model=model, render_episodes=0)
