from gym.envs.registration import registry, register, make, spec

register(
	id='CustomInvertedPendulum-v0',
	entry_point='gym_custominvertedpendulum.envs:CustomInvertedPendulumEnv',
	max_episode_steps=750,
	reward_threshold=600.0,
)
