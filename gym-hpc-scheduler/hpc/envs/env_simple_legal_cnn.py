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

MAX_QUEUE_SIZE = 64
MAX_JOBS_EACH_BATCH = 256
MIN_JOBS_EACH_BATCH = 1
MAX_WAIT_TIME = 12 * 60 * 60 # assume maximal wait time is 12 hours.
MAX_RUN_TIME = 12 * 60 * 60 # assume maximal runtime is 12 hours

THRESHOLD = (25600 * 100) # 10 processes

# each job has three features: wait_time, requested_node, runtime, machine states,
JOB_FEATURES = 4
DEBUG = False


class SimpleRandomLegalEnv(gym.Env):
    def __init__(self):  # do nothing and return. A workaround for passing parameters to the environment
        super(SimpleRandomLegalEnv, self).__init__()

        self.action_space = spaces.Discrete(MAX_QUEUE_SIZE)
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(JOB_FEATURES * MAX_QUEUE_SIZE,),
                                            dtype=np.float32)

        print("Initialize Simple HPC Env V9")
        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []

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

    def my_init(self, workload_file = ''):
        print ("loading workloads from dataset:", workload_file)
        self.loads = Workloads(workload_file)
        self.cluster = Cluster("Cluster", self.loads.max_nodes, self.loads.max_procs/self.loads.max_nodes)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sjf_score(self, job):
        run_time = job.run_time
        if job.job_id == 0:
            return sys.maxsize
        return run_time

    def fcfs_score(self, job):
        submit_time = job.submit_time
        if job.job_id == 0:
            return sys.maxsize
        return submit_time

    def reset(self):
        self.cluster.reset()
        self.loads.reset()

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.scheduled_logs = []
        self.scheduled_bsld = {}

        job_sequence_size = 64

        # randomly sample a sequence of jobs from workload (self.start_idx_last_reset + 1) % (self.loads.size() - 2 * job_sequence_size)
        self.start = self.np_random.randint(job_sequence_size, (self.loads.size() - 2 * job_sequence_size))
        self.start_idx_last_reset = self.start
        self.num_job_in_batch = job_sequence_size
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.next_arriving_job_idx = self.start + 1

        # schedule the sequence of jobs using the best heuristic algorithm.
        self.bsld_algo_dict = {}
        while True:
            get_this_job_scheduled = False

            self.job_queue.sort(key=lambda j: self.sjf_score(j))

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
        self.running_jobs = []
        self.visible_jobs = []
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.next_arriving_job_idx = self.start + 1

        # of course, we can schedule the first job in the system
        obs = self.build_observation()
        return obs

    def build_observation(self):
        vector = np.zeros((MAX_QUEUE_SIZE) * JOB_FEATURES, dtype=float)

        node_avail = 0.0
        for i in range(0, self.cluster.total_node):
            if self.cluster.all_nodes[i].is_free:
                node_avail += 1.0
            else:
                running_job_id = self.cluster.all_nodes[i].running_job_id
                running_job = None
                for _j in self.running_jobs:
                    if _j.job_id == running_job_id:
                        running_job = _j
                        break
                remainded = running_job.scheduled_time + running_job.run_time - self.current_timestamp
                node_avail += max(self.loads.max_exec_time - remainded, 0) / self.loads.max_exec_time
        node_avail_normal = (node_avail / self.cluster.total_node)

        self.job_queue.sort(key=lambda job: self.sjf_score(job))
        self.visible_jobs = []
        for i in range(0, MAX_QUEUE_SIZE - 1):
            if i < len(self.job_queue):
                # Check whether this job is possible to run.
                if self.cluster.can_allocated(self.job_queue[i]):
                    self.visible_jobs.append(self.job_queue[i])
            else:
                break
        
        assert len(self.visible_jobs) > 0
        random.shuffle(self.visible_jobs)
        # self.visible_jobs.sort(key=lambda j: self.fcfs_score(j))

        for i in range(0, MAX_QUEUE_SIZE - 1):
            if i < len(self.visible_jobs):
                job = self.visible_jobs[i]
                submit_time = job.submit_time
                request_processors = job.request_number_of_processors
                request_time = job.request_time
                # run_time = job.run_time
                wait_time = self.current_timestamp - submit_time

                normalized_wait_time = min(float(wait_time) / float(MAX_WAIT_TIME), 1)
                normalized_run_time = min(float(request_time) / float(self.loads.max_exec_time), 1)
                normalized_request_nodes = min(float(request_processors) / float(self.loads.max_procs), 1)

                vector[i*JOB_FEATURES:(i+1)*JOB_FEATURES] = [normalized_wait_time, normalized_run_time, normalized_request_nodes, node_avail_normal]
            else:
                vector[i*JOB_FEATURES:(i+1)*JOB_FEATURES] = [0,1,1,node_avail_normal]
        vector[(MAX_QUEUE_SIZE - 1)*JOB_FEATURES:MAX_QUEUE_SIZE*JOB_FEATURES] = [0,1,1,node_avail_normal]
        return np.reshape(vector, [-1, (MAX_QUEUE_SIZE) * JOB_FEATURES])

    def step(self, a):
        '''we have to make sure the state returned by step() always and only contains valid jobs to schedule'''
        action = a[0]

        if action >= len(self.visible_jobs):  
            # this is illegal action; it does not consume visible jobs; just move forward the time.
            if not self.running_jobs:  # there are no running jobs
                next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
                next_resource_release_machines = []
            else:
                self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
                next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                next_resource_release_machines = self.running_jobs[0].allocated_machines

            if self.next_arriving_job_idx < self.last_job_in_batch \
                    and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                assert self.current_timestamp <= self.loads[self.next_arriving_job_idx].submit_time
                self.current_timestamp = self.loads[self.next_arriving_job_idx].submit_time
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                if self.running_jobs:
                    # we are seeing the last job. 
                    assert self.current_timestamp <= next_resource_release_time
                    self.current_timestamp = next_resource_release_time
                    self.cluster.release(next_resource_release_machines)
                    self.running_jobs.pop(0)  # remove the first running job
        else:
            # consume a legal job, has to make sure we have enough legal jobs for the next state
            job_for_scheduling = self.visible_jobs[action]
            assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
            job_for_scheduling.scheduled_time = self.current_timestamp
            job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                            job_for_scheduling.request_number_of_processors)
            self.running_jobs.append(job_for_scheduling)
            self.scheduled_logs.append(job_for_scheduling)
            _tmp = max(1.0, (float(
                    job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)
                                /
                                max(job_for_scheduling.run_time, 10)))
            self.scheduled_bsld[job_for_scheduling.job_id] = (_tmp / self.num_job_in_batch)
            self.job_queue.remove(job_for_scheduling)  # remove the job from job queue
            
            # make sure there are legal jobs for the next state
            legal_job_exist = False
            for _j in self.job_queue:
                if self.cluster.can_allocated(_j):
                    legal_job_exist = True
                    break
            while not legal_job_exist:
                # move forward to get new jobs or finish running jobs
                if not self.running_jobs:  # there are no running jobs
                    next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
                    next_resource_release_machines = []
                else:
                    self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
                    next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                    next_resource_release_machines = self.running_jobs[0].allocated_machines

                if self.next_arriving_job_idx < self.last_job_in_batch \
                        and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                    assert self.current_timestamp <= self.loads[self.next_arriving_job_idx].submit_time
                    self.current_timestamp = self.loads[self.next_arriving_job_idx].submit_time
                    self.job_queue.append(self.loads[self.next_arriving_job_idx])
                    self.next_arriving_job_idx += 1
                else:
                    if self.running_jobs:
                        assert self.current_timestamp <= next_resource_release_time
                        self.current_timestamp = next_resource_release_time
                        self.cluster.release(next_resource_release_machines)
                        self.running_jobs.pop(0)  # remove the first running job.
                    else:
                        if not self.job_queue:
                            break;
                for _j in self.job_queue:
                    if self.cluster.can_allocated(_j):
                        legal_job_exist = True
                        break

        done = True
        for i in range(self.start, self.last_job_in_batch):
            if self.loads[i].scheduled_time == -1:  # have at least one job in the batch who has not been scheduled
                done = False
                break
        if done:
            algo1 = 0.0
            mine = 0.0
            for _job in self.scheduled_logs:
                algo1 += (self.bsld_algo_dict[_job.job_id])
                mine += (self.scheduled_bsld[_job.job_id])
            return [None, (algo1 - mine), True, None]
        else:
            obs = self.build_observation()
            return [obs, 0, False, None]