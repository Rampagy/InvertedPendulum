import multiprocessing as mp
import run_episodes as re
import NeuralNetwork as nn
import numpy as np
import gym

# Define an output queue
max_log = mp.Queue()

# create environment
env = gym.make('CartPole-v1')

# create model
model = nn.Control_Model(input_len=len(env.observation_space.low),
                         output_len=env.action_space.n)

for i in range(20):
    print('a')
    # Setup a list of processes that we want to run, one for each core
    processes = [mp.Process(target=re.run_env, args=(20, model, env, max_log))
                            for x in range(mp.cpu_count())]

    print('b')
    # Run processes
    for p in processes:
        p.start()

    print('c')
    # Exit the completed processes
    for p in processes:
        p.join()

    print('d')
    # Get process results from the output queue
    results = [max_log.get() for p in processes]

    max_key = -99999
    # train on the max score
    for score, actions, observs in results:
        if score > max_key:
            max_key = score
            max_actions = actions
            max_observs = observs

    print(max_observs.shape)
    print(max_observs[0].shape)
    model.train_game(max_observs, max_actions)

model.save_model()
