from gym.envs.registration import register

# DWA envs
register(
    id="dwa_param_continuous_laser-v0",
    entry_point="envs.parameter_tuning_envs:DWAParamContinuousLaser"
)

register(
    id="dwa_param_continuous_costmap-v0",
    entry_point="envs.parameter_tuning_envs:DWAParamContinuousCostmap"
)

# DWA planner assisted motion controller
register(
    id="motion_control_continuous_laser-v0",
    entry_point="envs.motion_control_envs:MotionControlContinuousLaser"
)

register(
    id="motion_control_continuous_costmap-v0",
    entry_point="envs.motion_control_envs:MotionControlContinuousCostmap"
)