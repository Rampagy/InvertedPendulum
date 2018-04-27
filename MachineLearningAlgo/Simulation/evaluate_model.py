import gym
#import NeuralNetwork as nn
import numpy as np


def EvalModel(model, env, eval_episodes, render_episodes):
    cumulative_reward = 0

    for i in range(eval_episodes):
        done = False

        observation = env.reset()

        while not done:
            if i < render_episodes:
                # render for viewing experience
                env.render()

            action = model.predict_move(np.asarray(observation).reshape((1, len(env.observation_space.low))))

            # use action to make a move
            observation, reward, done, info = env.step(action)
            cumulative_reward += reward

    print('average score: {}'.format(cumulative_reward/eval_episodes))
    return cumulative_reward/eval_episodes



if __name__ == "__main__":
    # create model
    model = nn.Control_Model()
    EvalModel(model)
