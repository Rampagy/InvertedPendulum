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
        trained_threshold=400, train_episodes=100, eval_episodes=15,
        render_episodes=0, save_inter=15):

    if enable_video:
        env = gym.wrappers.Monitor(env, directory=vid_dir, force=False, resume=True)

    train_count = 0
    eval_score = em.EvalModel(model, env, eval_episodes, render_episodes)

    # train to until above the threshold
    while(eval_score <= trained_threshold):
        max_reward = -np.inf

        # run a segment of X 'games' and train off of the max score
        for i in range(train_episodes):
            cumulative_reward = 0
            obs_log = []
            action_log = []
            done = False

            observation = np.asarray(env.reset()).reshape((1, len(env.observation_space.low)))


            while not done:
                # feed observation list into the model
                action = model.predict_move(observation)

                # keep a log of actions and observations
                obs_log += [observation]
                action_log += [action]

                # use action to make a move
                observation, reward, done, info = env.step(action)
                observation = np.asarray(observation).reshape((1, len(env.observation_space.low)))
                cumulative_reward += reward

            if cumulative_reward > max_reward:
                max_reward = cumulative_reward
                max_obs_log = np.squeeze(obs_log)
                max_action_log = action_log

        # train the dnn
        train_count += 1
        model.train_game(max_obs_log, max_action_log)

        # save model every multiple of save_inter
        if train_count%save_inter == 0:
            model.save_model()

        eval_score = em.EvalModel(model, env, eval_episodes, render_episodes)



    #print('{} training episodes'.format(train_count))
    # save the model for evaluation
    model.save_model()
    # Close the envirnment so the video can be written to
    env.close()

    return train_count, eval_score