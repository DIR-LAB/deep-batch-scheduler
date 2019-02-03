from gym.envs.registration import register

register(
        id='Scheduler-v0',
        entry_point='hpc.envs:HpcEnv',
)

register(
        id='Scheduler-cont-v0',
        entry_point='hpc.envs:HpcEnvCont',
)