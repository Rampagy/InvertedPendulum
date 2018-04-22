import gym
#import NeuralNetwork as nn
import numpy as np


def EvalModel(model, env, eval_episodes, render_episodes, obs_len):
    cumulative_reward = 0

    for i in range(eval_episodes):
        done = False
        obs_img = np.zeros((1, obs_len))

        observation = env.reset()

        while not done:
            if i < render_episodes:
                # render for viewing experience
                env.render()

            obs_img = np.roll(obs_img, 1, axis=0)
            obs_img[0, :] = observation
            reshaped = obs_img.reshape((1, obs_len))

            action = model.predict_move(reshaped, uniform=True)

            # use action to make a move
            observation, reward, done, info = env.step(action)
            cumulative_reward += reward

    print('average score: {}'.format(cumulative_reward/eval_episodes))
    return cumulative_reward/eval_episodes



if __name__ == "__main__":
    # create model
    model = nn.Control_Model()
    EvalModel(model)
