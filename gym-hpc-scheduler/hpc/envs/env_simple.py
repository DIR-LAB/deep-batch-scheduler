import numpy as np
import math
import sys
import os
import random

import gym
from gym import spaces
from gym.utils import seeding

from hpc.envs.job import Job
from hpc.envs.job import Workloads
from hpc.envs.cluster import Cluster

MAX_QUEUE_SIZE = 32
MAX_JOBS_EACH_BATCH = 256
MIN_JOBS_EACH_BATCH = 1
MAX_MACHINE_SIZE = 256
MAX_WAIT_TIME = 12 * 60 * 60 # assume maximal wait time is 12 hours.
MAX_RUN_TIME = 12 * 60 * 60 # assume maximal runtime is 12 hours

SEED = 42

# each job has three features: submit_time, request_number_of_processors, request_time/run_time,
JOB_FEATURES = 3
DEBUG = False


class SimpleHPCEnv(gym.Env):
    def __init__(self):  # do nothing and return. A workaround for passing parameters to the environment
        super(SimpleHPCEnv, self).__init__()

        self.action_space = spaces.Discrete(MAX_QUEUE_SIZE + 1)
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(JOB_FEATURES * (MAX_QUEUE_SIZE + 1),),
                                            dtype=np.float32)

        print("Initialize Simple HPC Env")

        # initialize random state used by the whole system.
        random.seed(SEED)

        self.job_queue = []
        self.running_jobs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.start_idx_last_reset = MAX_JOBS_EACH_BATCH

        self.loads = None
        self.cluster = None
        self.bsld_algo_dict = {}
        self.scheduled_logs = []
        self.scheduled_bsld = {}

        self.total_interactions = 0

    def my_init(self, workload_file = ''):
        print ("loading workloads from dataset:", workload_file)
        self.loads = Workloads(workload_file)
        self.cluster = Cluster("Cluster", self.loads.max_nodes, self.loads.max_procs/self.loads.max_nodes)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def f1_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        # request_time = job.request_time
        run_time = job.run_time
        if job.job_id == 0:
            return sys.maxsize
        return (np.log10(request_processors) * run_time + 870 * np.log10(submit_time))

    def reset(self):
        self.cluster.reset()
        self.loads.reset()

        self.job_queue = []
        self.running_jobs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.scheduled_logs = []
        self.scheduled_bsld = {}

        job_sequence_size = 8 * (1 + int(self.total_interactions / (16000 * 100))) # 100 epochs

        # randomly sample a sequence of jobs from workload random.randint(MAX_JOBS_EACH_BATCH, (self.loads.size() - 2 * job_sequence_size))
        self.start = (self.start_idx_last_reset + 1) % (self.loads.size() - 2 * job_sequence_size)
        self.start_idx_last_reset = self.start
        self.num_job_in_batch = job_sequence_size
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue[0] = self.loads[self.start]
        self.next_arriving_job_idx = self.start + 1

        # Generate some running jobs to randomly fill the cluster.
        q_workloads = []
        running_job_size = MAX_JOBS_EACH_BATCH
        for i in range(running_job_size):
            _job = self.loads[self.start - i - 1]
            req_num_of_processors = _job.request_number_of_processors
            runtime_of_job = _job.run_time
            job_tmp = Job()
            job_tmp.job_id = (-1 - i)  # to be different from the normal jobs; normal jobs have a job_id >= 0
            job_tmp.request_number_of_processors = req_num_of_processors
            job_tmp.run_time = runtime_of_job
            if self.cluster.can_allocated(job_tmp):
                self.running_jobs.append(job_tmp)
                # job_tmp.scheduled_time = max(0, (self.current_timestamp - random.randint(0, runtime_of_job)))
                job_tmp.scheduled_time = max(0, (self.current_timestamp - runtime_of_job/2))
                job_tmp.allocated_machines = self.cluster.allocate(job_tmp.job_id, job_tmp.request_number_of_processors)
                q_workloads.append(job_tmp)
            else:
                break

        # schedule the sequence of jobs using the best heuristic algorithm.
        self.bsld_algo_dict = {}
        while True:
            get_this_job_scheduled = False

            self.job_queue.sort(key=lambda j: self.f1_score(j))

            if self.cluster.can_allocated(self.job_queue[0]):
                job_for_scheduling = self.job_queue.pop(0)
                assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
                job_for_scheduling.scheduled_time = self.current_timestamp
                job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                            job_for_scheduling.request_number_of_processors)
                self.running_jobs.append(job_for_scheduling)
                _tmp = max(1.0, (float(
                    job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)
                                    /
                                    max(job_for_scheduling.run_time, 10)))
                self.bsld_algo_dict[job_for_scheduling.job_id] = (_tmp / self.num_job_in_batch)
                get_this_job_scheduled = True

            while not get_this_job_scheduled or not self.job_queue:
                if not self.running_jobs:  # there are no running jobs
                    next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
                    next_resource_release_machines = []
                else:
                    self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
                    next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                    next_resource_release_machines = self.running_jobs[0].allocated_machines

                if self.next_arriving_job_idx < self.last_job_in_batch \
                        and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                    self.job_queue.append(self.loads[self.next_arriving_job_idx])
                    self.current_timestamp = self.loads[self.next_arriving_job_idx].submit_time
                    self.next_arriving_job_idx += 1
                    break
                else:
                    if not self.running_jobs:
                        break
                    self.current_timestamp = next_resource_release_time
                    self.cluster.release(next_resource_release_machines)
                    self.running_jobs.pop(0)  # remove the first running job.

            done = True
            for i in range(self.start, self.last_job_in_batch):
                if self.loads[i].scheduled_time == -1:  # have at least one job in the batch who has not been scheduled
                    done = False
                    break
            if done:
                break

        # reset again
        self.cluster.reset()
        self.loads.reset()

        self.job_queue = []
        for i in range(0, MAX_QUEUE_SIZE):
            self.job_queue.append(Job())
        self.pending_job_queue = []

        self.running_jobs = []
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue[0] = self.loads[self.start]
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.next_arriving_job_idx = self.start + 1

        # use the same jobs to fill the cluster.
        for job_tmp in q_workloads:
            self.running_jobs.append(job_tmp)
            job_tmp.allocated_machines = self.cluster.allocate(job_tmp.job_id, job_tmp.request_number_of_processors)

        obs = self.build_observation()
        return obs

    def build_observation(self):
        self.job_queue.sort(key=lambda job: self.f1_score(job))

        vector = np.zeros((MAX_QUEUE_SIZE + 1) * JOB_FEATURES, dtype=float)

        for i in range(0, MAX_QUEUE_SIZE):
            if i < len(self.job_queue):
                job = self.job_queue[i]
                submit_time = job.submit_time
                request_processors = job.request_number_of_processors
                # request_time = job.request_time
                run_time = job.run_time
                wait_time = self.current_timestamp - submit_time
                normalized_wait_time = min(float(wait_time) / float(MAX_WAIT_TIME), 1)
                # normalized_request_time = min(float(request_time) / float(MAX_RUN_TIME), 1)
                normalized_run_time = min(float(run_time) / float(MAX_RUN_TIME), 1)
                normalized_request_nodes = min(float(request_processors) / float(self.loads.max_procs), 1)
                vector[i * JOB_FEATURES:(i+1)*JOB_FEATURES] = [normalized_wait_time, normalized_run_time, normalized_request_nodes]
            else:
                vector[i * JOB_FEATURES:(i+1)*JOB_FEATURES] = [0,0,0]
            

        cpu_avail = 0.0
        for i in range(0, MAX_MACHINE_SIZE):
            if self.cluster.all_nodes[i].is_free:
                cpu_avail += 1.0
            else:
                running_job_id = self.cluster.all_nodes[i].running_job_id
                running_job = None
                for _j in self.running_jobs:
                    if _j.job_id == running_job_id:
                        running_job = _j
                        break

                remainded = running_job.scheduled_time + running_job.run_time - self.current_timestamp
                cpu_avail += max(MAX_RUN_TIME - remainded, 0) / MAX_RUN_TIME

        vector[MAX_QUEUE_SIZE * JOB_FEATURES : (MAX_QUEUE_SIZE + 1) * JOB_FEATURES] = [(cpu_avail / MAX_MACHINE_SIZE), 0, 0]

        return np.reshape(vector, [-1, (MAX_QUEUE_SIZE + 1) * JOB_FEATURES])

    def step(self, a):
        # action is a legal job ready for scheduling.
        action = a[0]
        get_this_job_scheduled = False
        job_for_scheduling = None
        job_for_scheduling_index = -1
        self.total_interactions += 1

        # if action is the last item, this means no scheduling, just moving forward the time.
        if action == MAX_QUEUE_SIZE:
            get_this_job_scheduled = False
        else:
            if action >= len(self.job_queue):  # this is illegal action
                action = (len(self.job_queue) - 1)
            job_for_scheduling = self.job_queue[action]
            job_for_scheduling_index = action

            if self.cluster.can_allocated(job_for_scheduling):
                assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
                job_for_scheduling.scheduled_time = self.current_timestamp
                job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                              job_for_scheduling.request_number_of_processors)
                self.running_jobs.append(job_for_scheduling)
                self.scheduled_logs.append(job_for_scheduling)
                get_this_job_scheduled = True
                _tmp = max(1.0, (float(
                        job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)
                                 /
                                 max(job_for_scheduling.run_time, 10)))
                self.scheduled_bsld[job_for_scheduling.job_id] = (_tmp / self.num_job_in_batch)
                self.job_queue.pop(job_for_scheduling_index)  # remove the job from job queue

        while not get_this_job_scheduled or not self.job_queue:
            if not self.running_jobs:  # there are no running jobs
                next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
                next_resource_release_machines = []
            else:
                self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
                next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                next_resource_release_machines = self.running_jobs[0].allocated_machines

            if self.next_arriving_job_idx < self.last_job_in_batch \
                    and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                
                self.current_timestamp = self.loads[self.next_arriving_job_idx].submit_time
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
                break
            else:
                if not self.running_jobs:
                    break
                self.current_timestamp = next_resource_release_time
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.

        obs = self.build_observation()

        done = True
        for i in range(self.start, self.last_job_in_batch):
            if self.loads[i].scheduled_time == -1:  # have at least one job in the batch who has not been scheduled
                done = False
                break
        if done:
            algo = 0.0
            mine = 0.0
            for _job in self.scheduled_logs:
                algo += (self.bsld_algo_dict[_job.job_id])
                mine += (self.scheduled_bsld[_job.job_id])

            # GPU-3
            '''
            if mine <= 0.9 * algo:
                return [obs, 1000 * (algo - mine), True, None]  # a purely good case indicates huge plus
            elif mine < algo:
                return [obs, 100 * (algo - mine), True, None]  # a good case indicates big plus
            else:
                return [obs, (algo - mine), True, None]    # a normal case
            '''

            # GPU-1
            return [obs, (algo - mine), True, None]
            
            '''
            # GPU-2
            if mine < 0.95 * algo:
                return [obs, 1, True, None]
            elif 0.95 * algo <= mine < 1.05 * algo:
                return [obs, 0, True, None]
            else:
                return [obs, -1, True, None]
            '''
        else:
            return [obs, 0, False, None]