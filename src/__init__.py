from gym.envs.registration import register

register(
    id='linear_space_env-v0',
    entry_point='orbitModeControl.envs:linear_environment_lib',
)
