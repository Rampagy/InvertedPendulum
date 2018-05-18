from gym.envs.registration import registry, register, make, spec

register(
    id='CustomInvertedPendulum-v0',
    entry_point='gym_custominvertedpendulum.envs:CustomInvertedPendulumEnv',
    max_episode_steps=750,
    reward_threshold=550.0,
)

register(
    id='DampingPendulum-v0',
    entry_point='gym_custominvertedpendulum.envs:DampingPendulumEnv',
    max_episode_steps=750,
    reward_threshold=550.0,
)
