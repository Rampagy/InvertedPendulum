import multiprocessing as mp
import numpy as np
import run_episodes as re
import NeuralNetwork as nn
import gym

# Define an output queue
max_log = mp.Queue()
lock = mp.Lock()

# create environment
env = gym.make('CartPole-v1')

# create model
model = nn.Control_Model(input_len=len(env.observation_space.low),
                         output_len=env.action_space.n)

for i in range(20):

    re.run_env(80, model, env, max_log, lock)

    # Get process results from the output queue
    results = [max_log.get()]

    re.run_env(80, model, env, max_log, lock)

    results += [max_log.get()]

    max_key = -99999
    # train on the max score
    for score, actions, observs in results:
        if score > max_key:
            max_key = score
            max_actions = actions
            max_observs = observs

    #print(max_actions)
    print(max_observs.shape)
    print(max_observs[0].shape)
    model.train_game(max_observs, max_actions)

model.save_model()
