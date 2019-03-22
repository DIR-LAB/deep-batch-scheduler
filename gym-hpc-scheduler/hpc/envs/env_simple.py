import numpy as np
import math
import sys
import os
import random

import gym
from gym import spaces

from hpc.envs.job import Job
from hpc.envs.job import Workloads
from hpc.envs.cluster import Cluster

# HPC Batch Scheduler Simulation.
# 
# Created by Dong Dai. Licensed on the same terms as the rest of OpenAI Gym.

MAX_QUEUE_SIZE = 35
MAX_JOBS_EACH_BATCH = 2 * MAX_QUEUE_SIZE
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

        self.action_space = spaces.Discrete(MAX_QUEUE_SIZE)
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(JOB_FEATURES * (MAX_QUEUE_SIZE + 1),),
                                            dtype=np.float32)

        print("Initialize HPC Env Job")

        # initialize random state used by the whole system.
        random.seed(SEED)

        self.job_queue = []
        for i in range(0, MAX_QUEUE_SIZE):
            self.job_queue.append(Job())

        self.running_jobs = []
        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.schedule_logs = []
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0

        self.loads = None
        self.cluster = None
        self.bsld_fcfs_dict = {}
        self.scheduled_logs = []
        self.scheduled_bsld = {}

    def my_init(self, workload_file = ''):
        print ("loading workloads from dataset:", workload_file)
        self.loads = Workloads(workload_file)
        self.cluster = Cluster("Cluster", self.loads.max_nodes, self.loads.max_procs/self.loads.max_nodes)

    def reset(self):
        self.cluster.reset()
        self.loads.reset()

        self.job_queue = []
        for i in range(0, MAX_QUEUE_SIZE):
            self.job_queue.append(Job())

        self.running_jobs = []
        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.schedule_logs = []
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.scheduled_logs = []
        self.scheduled_bsld = {}

        # randomly sample a sequence of jobs from workload
        self.start = random.randint(MAX_JOBS_EACH_BATCH, (self.loads.size() - 2 * MAX_JOBS_EACH_BATCH))
        self.num_job_in_batch = random.randint(MAX_JOBS_EACH_BATCH, MAX_JOBS_EACH_BATCH)
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue[0] = self.loads[self.start]
        self.next_arriving_job_idx = self.start + 1

        # Generate some running jobs to randomly fill the cluster.
        q_workloads = []
        running_job_size = random.randint(MIN_JOBS_EACH_BATCH, MAX_JOBS_EACH_BATCH)
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
                # assume job was randomly generated
                # job_tmp.scheduled_time = max(0, (self.current_timestamp - random.randint(0, runtime_of_job)))
                job_tmp.scheduled_time = max(0, (self.current_timestamp - runtime_of_job/2))
                job_tmp.allocated_machines = self.cluster.allocate(job_tmp.job_id, job_tmp.request_number_of_processors)
                q_workloads.append(job_tmp)
            else:
                break

        # schedule the sequence of jobs using FCFS. This would be the standard references for this sequence.
        # v2: schedule the sequence of jobs using shortest job first.
        self.bsld_fcfs_dict = {}
        while True:
            # self.job_queue.sort(key=lambda j: (j.submit_time))
            self.job_queue.sort(key=lambda j: (j.run_time))
            get_this_job_scheduled = False

            for i in range(0, MAX_QUEUE_SIZE):
                if self.job_queue[i].job_id == 0:
                    continue
                job_for_scheduling = self.job_queue[i]
                job_for_scheduling_index = i

                if self.cluster.can_allocated(job_for_scheduling):
                    assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
                    job_for_scheduling.scheduled_time = self.current_timestamp
                    job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                                  job_for_scheduling.request_number_of_processors)
                    self.running_jobs.append(job_for_scheduling)
                    _tmp = max(1.0, (float(
                        job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)
                                     /
                                     max(job_for_scheduling.run_time, 10)))
                    self.bsld_fcfs_dict[job_for_scheduling.job_id] = (_tmp / self.num_job_in_batch)
                    get_this_job_scheduled = True
                    self.job_queue[i] = Job()  # remove the job from job queue
                    break

            while not get_this_job_scheduled or self._is_job_queue_empty():
                if not self.running_jobs:  # there are no running jobs
                    next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
                    next_resource_release_machines = []
                else:
                    self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
                    next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                    next_resource_release_machines = self.running_jobs[0].allocated_machines

                if self.next_arriving_job_idx < self.last_job_in_batch \
                        and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time \
                        and not self._is_job_queue_full():

                    for i in range(0, MAX_QUEUE_SIZE):
                        if self.job_queue[i].job_id == 0:
                            self.job_queue[i] = self.loads[self.next_arriving_job_idx]
                            # current timestamp may be larger than next_arriving_job's submit time because job queue was
                            # full and we move forward to release resources.
                            self.current_timestamp = max(self.current_timestamp,
                                                         self.loads[self.next_arriving_job_idx].submit_time)
                            self.next_arriving_job_idx += 1
                            break
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

        self.running_jobs = []
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue[0] = self.loads[self.start]
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.next_arriving_job_idx = self.start + 1

        # use the same jobs to fill the cluster.

        for job_tmp in q_workloads:
            self.running_jobs.append(job_tmp)
            job_tmp.allocated_machines = self.cluster.allocate(job_tmp.job_id, job_tmp.request_number_of_processors)


        if DEBUG:
            print("Reset:%s, %d, %s" %
                  self.loads[self.start],
                  self.num_job_in_batch,
                  self.loads[self.last_job_in_batch])

        obs = self.build_observation()
        return obs

    def build_observation(self):
        sq = int(math.ceil(math.sqrt(MAX_QUEUE_SIZE)))
        job_queue_row = sq

        vector = np.zeros((job_queue_row, sq, JOB_FEATURES), dtype=float)
        # self.job_queue.sort(key=lambda j: j.submit_time, reverse=True)

        for i in range(0, MAX_QUEUE_SIZE):
            job = self.job_queue[i]

            submit_time = job.submit_time
            request_processors = job.request_number_of_processors
            request_time = job.request_time
            run_time = job.run_time
            # not used for now
            #user_id = job.user_id
            #group_id = job.group_id
            #executable_number = job.executable_number
            #queue_number = job.queue_number

            if job.job_id == 0:
                wait_time = 0
            else:
                wait_time = self.current_timestamp - submit_time
            normalized_wait_time = min(float(wait_time) / float(MAX_WAIT_TIME), 1)
            # normalized_request_time = min(float(request_time) / float(MAX_RUN_TIME), 1)
            normalized_run_time = min(float(run_time) / float(MAX_RUN_TIME), 1)
            normalized_request_nodes = min(float(request_processors) / float(self.loads.max_procs), 1)

            vector[int(i / sq), int(i % sq)] = [normalized_wait_time, normalized_run_time, normalized_request_nodes]

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

        vector[int(MAX_QUEUE_SIZE / sq), int(MAX_QUEUE_SIZE % sq)] = [(cpu_avail / MAX_MACHINE_SIZE), 0, 0]

        return np.reshape(vector, [-1, (MAX_QUEUE_SIZE + 1) * JOB_FEATURES])

    def _is_job_queue_empty(self):
        return all(v.job_id == 0 for v in self.job_queue)

    def _is_job_queue_full(self):
        return all(v.job_id > 0 for v in self.job_queue)

    def _job_queue_size(self):
        size = 0
        for i in range(0, MAX_QUEUE_SIZE):
            if self.job_queue[i].job_id > 0:
                size += 1
        return size

    def step(self, a):
        # action is a legal job ready for scheduling.
        action = a[0]

        # if current job is illegal, map it to a nearby legal job.
        if self.job_queue[action].job_id == 0:
            picked_job = - (2 * MAX_JOBS_EACH_BATCH)
            for i in range(0, action):
                if self.job_queue[action - i - 1].job_id != 0:
                    picked_job = (action - 1 - i)
                    break
            for i in range(action + 1, MAX_QUEUE_SIZE):
                if self.job_queue[i].job_id != 0:
                    if (i - action) < (action - picked_job):
                        picked_job = i
                    break
            action = picked_job

        assert self.job_queue[action].job_id != 0
        #if self.job_queue[action].job_id == 0:  # this job should be legal.
        #    obs = self.build_observation()
        #    return [obs, 0, True, None]

        job_for_scheduling = self.job_queue[action]
        job_for_scheduling_index = action

        get_this_job_scheduled = False

        if self.cluster.can_allocated(job_for_scheduling):
            assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
            job_for_scheduling.scheduled_time = self.current_timestamp
            if DEBUG:
                print("In step, schedule a job, ", job_for_scheduling, " with free nodes: ", self.cluster.free_node)
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
            self.job_queue[job_for_scheduling_index] = Job()  # remove the job from job queue

        while not get_this_job_scheduled or self._is_job_queue_empty():
            if not self.running_jobs:  # there are no running jobs
                next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
                next_resource_release_machines = []
            else:
                self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
                next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                next_resource_release_machines = self.running_jobs[0].allocated_machines

            if self.next_arriving_job_idx < self.last_job_in_batch \
                    and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time \
                    and not self._is_job_queue_full():

                for i in range(0, MAX_QUEUE_SIZE):
                    if self.job_queue[i].job_id == 0:
                        self.job_queue[i] = self.loads[self.next_arriving_job_idx]
                        # current timestamp may be larger than next_arriving_job's submit time because job queue was
                        # full and we move forward to release resources.
                        self.current_timestamp = max(self.current_timestamp,
                                                     self.loads[self.next_arriving_job_idx].submit_time)
                        self.next_arriving_job_idx += 1
                        break
                break
            else:
                if not self.running_jobs:
                    break
                self.current_timestamp = next_resource_release_time
                self.cluster.release(next_resource_release_machines)
                removed_job = self.running_jobs.pop(0)  # remove the first running job.
                if DEBUG:
                    print("In step, release a job, ", removed_job, " generated free nodes: ", self.cluster.free_node)

        obs = self.build_observation()

        done = True
        for i in range(self.start, self.last_job_in_batch):
            if self.loads[i].scheduled_time == -1:  # have at least one job in the batch who has not been scheduled
                done = False
                break
        if done:
            fcfs = 0.0
            mine = 0.0
            for _job in self.scheduled_logs:
                fcfs += (self.bsld_fcfs_dict[_job.job_id])
                mine += (self.scheduled_bsld[_job.job_id])

            # GPU-3
            '''
            if mine <= 0.9 * fcfs:
                return [obs, 1000 * (fcfs - mine), True, None]  # a purely good case indicates huge plus
            elif mine < fcfs:
                return [obs, 100 * (fcfs - mine), True, None]  # a good case indicates big plus
            else:
                return [obs, (fcfs - mine), True, None]    # a normal case
            '''

            # GPU-1
            '''
            return [obs, (fcfs - mine), True, None]
            '''

            # GPU-2
            if mine < 0.95 * fcfs:
                return [obs, 1, True, None]
            elif 0.95 * fcfs <= mine < 1.05 * fcfs:
                return [obs, 0, True, None]
            else:
                return [obs, -1, True, None]

        else:
            return [obs, 0, False, None]