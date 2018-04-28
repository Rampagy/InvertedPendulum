#import ../NeuralNetwork as nn
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
        trained_threshold=400, eval_episodes=15, train_depth=20,
        render_episodes=0, save_inter=15, train_episodes=100):

    if enable_video:
        env = gym.wrappers.Monitor(env, directory=vid_dir, force=False, resume=True)

    episode_count = 0
    train_count = 0
    eval_score = em.EvalModel(model, env, eval_episodes, render_episodes)

    # train to until above the threshold
    while(eval_score <= trained_threshold):
        max_reward = -np.inf
        done = False
        train = False
        max_obs_log = np.empty((1, 4))
        max_action_log = np.empty((1, 1))

        for i in range(train_episodes):
            observation = np.asarray(env.reset()).reshape((1, len(env.observation_space.low)))
            episode_count += 1

            obs_log = np.empty((1, 4))
            action_log = np.empty((1, 1))

            while not done:
                # feed observation list into the model
                action = model.predict_move(observation)

                # keep a log of actions and observations
                obs_log = np.append(obs_log, observation, axis=0)
                action_log = np.append(action_log, np.asarray(action).reshape((1, 1)), axis=0)

                # use action to make a move
                observation, reward, done, info = env.step(action)
                observation = np.asarray(observation).reshape((1, len(env.observation_space.low)))

                if reward > 0:
                    # only append as long as the log is or clip at 'train_depth'
                    lookback = np.minimum(obs_log.shape[0]-1, train_depth)

                    # If this move created a reward of +0 (or more) add it and
                    # the last 'train_depth' moves to the train log.
                    # This rewards behavior that leads to points AND moves that keep points
                    max_obs_log = np.append(max_obs_log, obs_log[-lookback:, :], axis=0)
                    max_action_log = np.append(max_action_log, action_log[-lookback:, :], axis=0)
                    train = True

        if train:
            # train the dnn
            print('TRAIN')
            model.train_game(max_obs_log, max_action_log)


        train_count += 1
        # save model every multiple of save_inter
        if train_count%save_inter == 0:
            model.save_model()

        eval_score = em.EvalModel(model, env, eval_episodes, render_episodes)



    print('{} training episodes'.format(episode_count))
    # save the model for evaluation
    model.save_model()
    # Close the envirnment so the video can be written to
    env.close()

    return train_count, eval_score
