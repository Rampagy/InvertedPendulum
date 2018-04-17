import multiprocessing as mp
import numpy as np
import random
import gym

# Define an output queue
max_log = mp.Queue()

# define an example function
def run_env(episodes, env, max_log):
    max_reward = -99999999
    np.random.seed(random.randint(0, 10000))

    for i in range(episodes):
        cumulative_reward = 0
        obs_log = []
        action_log = []
        done = False

        # reset env
        observation = env.reset()

        while not done:
            # render for viewing experience
            env.render()

            # pick a random action
            rand_weight = random.random()
            occurance = np.random.multinomial(1, [rand_weight,1-rand_weight])
            action = np.argmax(occurance)

            # keep a log of actions and observations
            obs_log += [observation]
            action_log += [action]

            # use action to make a move
            observation, reward, done, info = env.step(action)
            cumulative_reward += reward

        if cumulative_reward > max_reward:
            max_reward = cumulative_reward
            max_obs_log = obs_log
            max_action_log = action_log


    out = {max_reward : (action_log, max_action_log)}
    max_log.put(out)

# create environment
env = gym.make('CartPole-v1')

# Setup a list of processes that we want to run, one for each core
processes = [mp.Process(target=run_env, args=(100, env, max_log)) for x in range(mp.cpu_count())]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

# Get process results from the output queue
results = [max_log.get() for p in processes]

print(results)
