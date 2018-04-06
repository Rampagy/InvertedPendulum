import gym_custominvertedpendulum as gym
import NeuralNetwork as nn
import evaluate_model as em
import numpy as np

# setup parameters
env_name = 'CustomInvertedPendulum-v0'
vid_dir = 'Video/'
enable_video = False
historic_data_len = 10 # number of previous observation to feed into the nn
trained_threshold = 495 # reward threshold at which it stops training
train_episodes = 200 # number of episodes before training
eval_episodes = 15 # number of episode when evaluating
render_episodes = 1 # number of episodes to render when evaluating

env = gym.make(env_name)
if enable_video:
    env = gym.wrappers.Monitor(env, directory=vid_dir, force=False, resume=True)

obs_len = len(env.observation_space.low)
act_len = env.action_space.n

# create model
model = nn.Control_Model(obs_len*historic_data_len, act_len)

train_count = 0

# train to until above the threshold
while(em.EvalModel(model, env, eval_episodes, render_episodes, historic_data_len, obs_len) < trained_threshold):
    max_reward = -np.inf

    # run a segment of 200 'games' and train off of the max score
    for i in range(train_episodes):
        cumulative_reward = 0
        obs_log = []
        action_log = []
        done = False
        obs_img = np.zeros((historic_data_len, obs_len))

        observation = env.reset()

        while not done:
            obs_img = np.roll(obs_img, 1, axis=0)
            obs_img[0, :] = observation
            reshaped = obs_img.reshape((1, historic_data_len*obs_len))

            action = model.predict_move(reshaped)

            # if the model is not initialized, take a random action instead
            if action == None:
                action = env.action_space.sample()

            # keep a log of actions and observations
            obs_log += [reshaped.flatten()]
            action_log += [action]

            # use action to make a move
            observation, reward, done, info = env.step(action)
            cumulative_reward += reward

        if cumulative_reward > max_reward:
            max_reward = cumulative_reward
            max_obs_log = obs_log
            max_action_log = action_log

        print('Episode {} scored {}, max {}'.format(i, cumulative_reward, max_reward))


    # train the dnn
    train_count += 1
    model.train_game(max_obs_log, max_action_log)



print('{} training episodes'.format(train_count))
# save the model for evaluation
model.save_model()
# Close the envirnment so the video can be written to
env.close()
