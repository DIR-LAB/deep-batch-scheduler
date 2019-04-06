from gym.envs.registration import register

register(
        id='Scheduler-v0',
        entry_point='hpc.envs:HpcEnv',
)

register(
        id='Scheduler-cont-v0',
        entry_point='hpc.envs:HpcEnvCont',
)

register(
        id='Scheduler-v1',
        entry_point='hpc.envs:HpcEnvJob',
)

register(
        id='Scheduler-v3',
        entry_point='hpc.envs:HpcEnvJobLegal',
)

register(
        id='Scheduler-v4',
        entry_point='hpc.envs:SimpleHPCEnv',
)

register(
        id='Scheduler-v5',
        entry_point='hpc.envs:SimpleDirectHPCEnv',
)

register(
        id='Scheduler-v6',
        entry_point='hpc.envs:SimpleRandomHPCEnv',
)

register(
        id='Scheduler-v7',
        entry_point='hpc.envs:SimpleRandomShuffleHPCEnv',
)

register(
        id='Scheduler-v8',
        entry_point='hpc.envs:SimpleRandomCNNHPCEnv',
)