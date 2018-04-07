from gym.envs.registration import registry, register, make, spec

register(
    id='CustomInvertedPendulum-v0',
    entry_point='gym_custominvertedpendulum.envs:CustomInvertedPendulumEnv',
    max_episode_steps=750,
    reward_threshold=500.0,
)

register(
    id='CustomCartPole-v0',
    entry_point='gym_custominvertedpendulum.envs:CustomCartPoleEnv',
    max_episode_steps=750,
    reward_threshold=700.0,
)
