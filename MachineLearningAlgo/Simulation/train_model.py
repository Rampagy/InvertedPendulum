import evaluate_model as em
import numpy as np

'''
env                     # environment
model                   # tensorflow/tflearn model
vid_dir                 # place to save video
enable_video            # if video should be captured
trained_threshold       # reward threshold at which it stops training
train_episodes          # number of episodes before training
eval_episodes           # number of episode when evaluating
render_episodes         # number of episodes to render when evaluating
obs_len                 # length of the observations
save_inter              # number of intervals between saves
'''

def train_model(model, env, vid_dir='Video', enable_video=False,
        trained_threshold=400, eval_episodes=15, train_depth=50,
        render_episodes=0, save_inter=15, train_episodes=100):

    if enable_video:
        env = gym.wrappers.Monitor(env, directory=vid_dir, force=False, resume=True)

    episode_count = 0
    train_count = 0
    eval_score = em.EvalModel(model, env, eval_episodes, render_episodes)

    # train to until above the threshold
    while(eval_score <= trained_threshold):
        train_obs_log = np.empty((1, len(env.observation_space.low)))
        train_action_log = np.empty((1, 1))

        for i in range(train_episodes):
            done = False
            episode_count += 1

            obs_log = np.empty((1, len(env.observation_space.low)), dtype=np.float32)
            action_log = np.empty((1, 1), dtype=np.int32)
            reward_log = np.empty((1, 1), dtype=np.int32)

            observation = np.asarray(env.reset()).reshape((1, len(env.observation_space.low)))

            while not done:
                # feed observation list into the model
                action = model.predict_move(observation)

                # keep a log of actions and observations
                obs_log = np.append(obs_log, observation, axis=0)
                action_log = np.append(action_log, np.asarray(action).reshape((1, 1)), axis=0)

                # use action to make a move
                observation, reward, done, info = env.step(action)
                observation = np.asarray(observation).reshape((1, len(env.observation_space.low)))

                # keep a log of the reward
                reward_log = np.append(reward_log, np.asarray(reward).reshape((1, 1)), axis=0)

            # trim out the init value (np.empty) in the logs
            obs_log = obs_log[1:, :]
            action_log = action_log[1:, :]
            reward_log = reward_log[1:, :]

            # compute backprop data
            train_obs_log, train_action_log = filter_episode(obs_log, action_log, reward_log)

            # if there is train data
            if train_obs_log.shape[0] >= 2:
                model.train_game(train_obs_log, train_action_log)

        train_count += 1
        # save model every multiple of save_inter
        if train_count%save_inter == 0:
            model.save_model()

        eval_score = em.EvalModel(model, env, eval_episodes, render_episodes)

        # write to the log file
        with open("../Model/scores.txt", "a") as myfile:
            myfile.write(str(eval_score) + '\n')


    print('{} training episodes'.format(episode_count))
    # save the model for evaluation
    model.save_model()
    # Close the environment so the video can be written to
    env.close()

    return train_count, eval_score




def filter_episode(obs_log, action_log, reward_log):
    max_game_steps = 750
    step_count = 0
    forward_reward_steps_lim = 40
    early_end_steps_lim = 15

    log_len = action_log.shape[0]

    train_obs_log = np.empty((1, obs_log.shape[1]))
    train_action_log = np.empty((1, 1))

    for act, obs in zip(action_log, obs_log):
        x, x_dot, theta, theta_dot = obs

        # look forward X steps or until the end of the log
        reward_steps = np.minimum(log_len - (step_count+1), forward_reward_steps_lim)

        obs = np.asarray(obs).reshape(1, obs_log.shape[1])

        # if the game ended early discourage the last 'early_end_steps_lim' moves
        if (log_len < max_game_steps) and \
                (step_count+1 > log_len-early_end_steps_lim):

            # discourage network for letting the game end early
            act = np.asarray(int(not act)).reshape(1, action_log.shape[1])

            train_obs_log = np.append(train_obs_log, obs, axis=0)
            train_action_log = np.append(train_action_log, act, axis=0)

        # if it got a reward for its current position OR
        # if the current move results in a point within the next 'reward_steps' steps
        elif (reward_log[step_count] >= 0.5) or \
                (np.sum(reward_log[step_count:step_count+reward_steps]) >= 0.5):

            # convert action to numpy array
            act = np.asarray(act).reshape(1, action_log.shape[1])

            train_obs_log = np.append(train_obs_log, obs, axis=0)
            train_action_log = np.append(train_action_log, act, axis=0)


        step_count += 1

    # trim out the init value (np.empty) in the logs
    train_obs_log = train_obs_log[1:, :]
    train_action_log = train_action_log[1:, :]

    return train_obs_log, train_action_log
