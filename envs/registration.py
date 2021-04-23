from gym.envs.registration import register

# DWA envs
register(
    id="dwa_param_continuous-v0",
    entry_point="envs.parameter_tuning_envs:DWAParamContinuous"
)