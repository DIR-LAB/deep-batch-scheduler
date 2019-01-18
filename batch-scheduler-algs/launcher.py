import numpy
import tensorflow
import gym

env1 = gym.make('Scheduler-v0')
env1.env.init(kwargs={'workload_files': "./data/RICC-2010-2.swf"})
