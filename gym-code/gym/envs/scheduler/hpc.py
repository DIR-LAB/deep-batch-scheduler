import numpy as np
import math
import gym
from gym.utils import seeding

from job import Job
from job import Workloads

# HPC Batch Scheduler Simulation.
# 
# Created by Dong Dai. Licensed on the same terms as the rest of OpenAI Gym.

class Hpc(gym.Env):
    def __init__(self, workload_file):
        # workload_file = "./data/RICC-2010-2.swf"
        print ("loading workloads from dataset")
        self.workload_file = workload_file
        self.load = Workloads(workload_file)
        
        self.job_queue = []
        self.running_jobs = []
        self.start_timestamp = 0
        self.current_timestamp = 0
        self.workload_start = 0
        self.workload_size = 0
        self.start = 0
        self.size = 0
        self.current_point = 0
        self.longest_run_time = 0
        self.longest_average_delay = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        return 0

    def reset(self):
        return 0