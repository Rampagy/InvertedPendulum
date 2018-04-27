import sys
sys.path.append("../Shared")

import evaluate_model as em
import NeuralNetwork as nn
import gym_custominvertedpendulum as gym
from timeit import default_timer as timer

start = timer()

# create env to initialize model
env = gym.make('CustomInvertedPendulum-v0')
obs_len = len(env.observation_space.low)

# create model
model = nn.Control_Model(obs_len)

# show model
eval_score = em.EvalModel(model, env, 100, 10)

env.close()

'''
# create env to initialize model
env = gym.make('CustomInvertedPendulum_DisturbReject-v0')
obs_len = len(env.observation_space.low)
act_len = env.action_space.n

# show model
eval_score = em.EvalModel(model, env, 100, 0, obs_len)
'''

end = timer()
print(end - start)

env.close()
