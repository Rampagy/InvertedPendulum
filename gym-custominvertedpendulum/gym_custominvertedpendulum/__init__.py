from gym.envs.registration import registry, register, make, spec

register(
    id='CustomInvertedPendulum-v0',
    entry_point='gym_custominvertedpendulum.envs:CustomInvertedPendulumEnv',
    max_episode_steps=750,
    reward_threshold=550.0,
)

register(
    id='CustomCartPole-v0',
    entry_point='gym_custominvertedpendulum.envs:CustomCartPoleEnv',
    max_episode_steps=750,
    reward_threshold=700.0,
)

register(
    id='CustomInvertedPendulum_DisturbReject-v0',  # Disturbance rejection training environment
    entry_point='gym_custominvertedpendulum.envs:CustomInvertedPendulumDisturbRejectEnv',
    max_episode_steps=2000,
    reward_threshold=1500.0,
)
