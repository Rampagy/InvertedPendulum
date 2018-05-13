import evaluate_model as em
import numpy as np

'''
model                   # tensorflow/tflearn model
env                     # environment
vid_dir                 # place to save video
enable_video            # if video should be captured
trained_threshold       # reward threshold at which it stops training
eval_episodes           # number of episode when evaluating
render_episodes         # number of episodes to render when evaluating
save_inter              # number of intervals between saves
train_episodes          # number of games to run before training
'''

def train_model(model, env, vid_dir='Video', enable_video=False,
        trained_threshold=400, eval_episodes=15, render_episodes=0,
        save_inter=15, train_episodes=10, eval_iter=30):

    if enable_video:
        env = gym.wrappers.Monitor(env, directory=vid_dir, force=False, resume=True)

    episode_count = 0
    train_count = 0
    eval_score = em.EvalModel(model, env, eval_episodes, render_episodes)

    # train to until above the threshold
    while(eval_score <= trained_threshold):

        # train 'eval_iter' before evaulating if it got better
        for _ in range(eval_iter):
            for _ in range(train_episodes):
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

                # extract data for training
                train_obs_log, train_action_log = extract_trainable_data(obs_log, action_log, reward_log)

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
    # Close the environment so the video can be written to the disk
    env.close()

    return train_count, eval_score



def extract_trainable_data(obs_log, action_log, reward_log):
    railwidth = 1.42/2 - 0.065/2 # half the rail length minus half the width of the cart

    train_obs_log = np.empty((1, 4))
    train_act_log = np.empty((1, 1))

    prev_value = 0
    prev_act = np.zeros(shape=(1,1))
    prev_obs = np.zeros(shape=(1,4))

    for obs, act in zip(obs_log, action_log):
        obs = np.asarray([obs])
        act = np.asarray([act])

        value = np.cos(obs[0, 2]/2) + 0.5*abs(obs[0, 0])/railwidth

        # if the state got better from its previous action
        # or it is already in a good state
        if (value - prev_value > 0.05) or (value > 1):
            # train off of the previous action and state
            train_obs_log = np.append(train_obs_log, prev_obs, axis=0)
            train_act_log = np.append(train_act_log, prev_act, axis=0)

        prev_value = value
        prev_act = act
        prev_obs = obs

    if train_obs_log.shape[0] > 1:
        # trim out init value (np.empty)
        train_obs_log = train_obs_log[1:, :]
        train_act_log = train_act_log[1:, :]

    return train_obs_log, train_act_log
