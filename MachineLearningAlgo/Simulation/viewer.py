import evaluate_model as em
import NeuralNetwork as nn
import gym_custominvertedpendulum as gym
from timeit import default_timer as timer

start = timer()

# create env to initialize model
env = gym.make('CustomInvertedPendulum-v0')
obs_len = len(env.observation_space.low)
act_len = env.action_space.n

# create model
model = nn.Control_Model(obs_len, act_len)

# show model
eval_score = em.EvalModel(model, env, 10, 0, obs_len)

env.close()

# create env to initialize model
env = gym.make('CustomInvertedPendulum_DisturbReject-v0')
obs_len = len(env.observation_space.low)
act_len = env.action_space.n

# show model
eval_score = em.EvalModel(model, env, 10, 0, obs_len)


end = timer()
print(end - start)

env.close()