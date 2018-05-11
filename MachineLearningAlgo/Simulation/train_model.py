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
eval_iter               # number of train_episodes to 
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
            train_obs_log = np.empty((1, len(env.observation_space.low)))
            train_action_log = np.empty((1, 1))
            max_score = -np.inf

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

                if sum(reward_log) > max_score:
                    max_score = sum(reward_log)
                    train_obs_log = obs_log
                    train_action_log = action_log

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


