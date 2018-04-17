def run_env(episodes, model, env, max_log):
    import numpy as np
    import random
    import sys

    max_reward = -np.inf
    np.random.seed(random.randint(0, 20000))
    obs_len = len(env.observation_space.low)

    for i in range(episodes):
        cumulative_reward = 0
        obs_log = np.zeros((1, obs_len))
        action_log = []
        done = False

        # reset env
        observation = env.reset()

        while not done:
            # predict an action
            obs = np.reshape(observation, (-1, obs_len))
            print('e')
            sys.stdout.flush()
            action = model.predict_move(obs)
            print('f')
            sys.stdout.flush()

            # keep a log of actions and observations
            obs_log = np.append(obs_log, obs, axis=0)
            action_log += [action]

            # use action to make a move
            observation, reward, done, info = env.step(action)
            cumulative_reward += reward

        if cumulative_reward > max_reward:
            max_reward = cumulative_reward
            max_obs_log = obs_log[1:, :] # trim out init row (all zeros)
            max_action_log = action_log

    out = (max_reward, max_action_log, max_obs_log)
    max_log.put(out)
