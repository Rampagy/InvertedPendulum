import train_model as tm
import NeuralNetwork as nn
import gym_custominvertedpendulum as gym

# toggle between inverted pendulum and cartpole with increasing trained_threshold
# so that the model learns to flip the pole up and balance it once flipped up.

count = 0;
inverted_pend_eval_score = 0

#create env to initialize model
env = gym.make('CustomInvertedPendulum-v0')
obs_len = len(env.observation_space.low)
act_len = env.action_space.n

# create model
model = nn.Control_Model(obs_len, act_len)


# train until the inverted pendulum gets a score above 200
while (inverted_pend_eval_score <= 200):

    # on even numbers do cartpole
    if count%2==0:
        # practive balancing at the top
        env_name = 'CustomCartPole-v0'
        trained_threshold = 100
    # on odd numbers do inverted pendulum
    else:
        # practive flipping the pole up
        env_name = 'CustomInvertedPendulum-v0'
        # slowly demand that the inverted pendulum hold it longer and longer
        trained_threshold = 5 + count
        if trained_threshold > 600:
            trainted_threshold = 600

    env = gym.make(env_name)

    obs_len = len(env.observation_space.low)
    act_len = env.action_space.n

    train_count, eval_score = tm.train_model(env=env,
        trained_threshold=trained_threshold, model=model, obs_len=obs_len)

    if env_name == 'CustomInvertedPendulum-v0':
        inverted_pend_eval_score = eval_score

    count += 1
