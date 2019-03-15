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

MAX_QUEUE_SIZE = 16
MAX_JOBS_EACH_BATCH = 16
MIN_JOBS_EACH_BATCH = 1
MAX_MACHINE_SIZE = 256
MAX_WAIT_TIME = 12 * 60 * 60 # assume maximal wait time is 12 hours.
MAX_RUN_TIME = 12 * 60 * 60 # assume maximal runtime is 12 hours

SEED = 42

# each job has three features: submit_time, request_number_of_processors, request_time/run_time,
JOB_FEATURES = 3
DEBUG = False


class HpcEnvJobLegal(gym.Env):
    def __init__(self):  # do nothing and return. A workaround for passing parameters to the environment
        super(HpcEnvJobLegal, self).__init__()

        self.action_space = spaces.Discrete(MAX_QUEUE_SIZE)
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(JOB_FEATURES * (MAX_QUEUE_SIZE + MAX_MACHINE_SIZE),),
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
        self.start = random.randint(MAX_JOBS_EACH_BATCH, (self.loads.size() - MAX_JOBS_EACH_BATCH))
        self.num_job_in_batch = random.randint(MIN_JOBS_EACH_BATCH, MAX_JOBS_EACH_BATCH)
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue[0] = self.loads[self.start]
        self.next_arriving_job_idx = self.start + 1

        # Generate some running jobs to randomly fill the cluster.

        q_workloads = []
        running_job_size = MAX_JOBS_EACH_BATCH # random.randint(MAX_JOBS_EACH_BATCH, MAX_JOBS_EACH_BATCH)
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
            self.job_queue.sort(key=lambda j: (j.submit_time))
            # self.job_queue.sort(key=lambda j: (j.run_time))
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
                else:
                    # if there is no enough resource for current job, try to backfill using other jobs

                    # calculate the expected starting time of current job.
                    _needed_processors = job_for_scheduling.request_number_of_processors
                    _expected_start_time = self.current_timestamp
                    _extra_released_processors = 0

                    self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
                    _free_processors_ = self.cluster.free_node * self.cluster.num_procs_per_node
                    for _job in self.running_jobs:
                        _free_processors_ += len(_job.allocated_machines) * self.cluster.num_procs_per_node
                        released_time = _job.scheduled_time + _job.run_time
                        if _free_processors_ >= _needed_processors:
                            _expected_start_time = released_time
                            _extra_released_processors = _free_processors_ - _needed_processors
                            break
                    assert _free_processors_ >= _needed_processors

                    # find do we have other jobs that do not affect the _expected_start_time
                    for j in range(0, MAX_QUEUE_SIZE):
                        _schedule_it = False
                        _job = self.job_queue[j]
                        if _job.job_id == 0 or _job.job_id == job_for_scheduling_index:
                            continue
                        assert _job.scheduled_time == -1  # this job should never be scheduled before.
                        if not self.cluster.can_allocated(_job):
                            # if this job can not be allocated, skip to next job in the queue
                            continue

                        if (_job.run_time + self.current_timestamp) <= _expected_start_time:
                            # if i can finish earlier than the expected start time of job[i], schedule it.
                            _schedule_it = True
                        else:
                            # even this _job lasts longer, but it uses less than _extra processors, it does not affect current job too
                            if _job.request_number_of_processors < _extra_released_processors:
                                _schedule_it = True
                                # but, we have to update the _extra_released_processors if we took it
                                _extra_released_processors -= _job.request_number_of_processors

                        if _schedule_it:
                            _job.scheduled_time = self.current_timestamp
                            _job.allocated_machines = self.cluster.allocate(_job.job_id,
                                                                            _job.request_number_of_processors)
                            self.running_jobs.append(_job)
                            _tmp = max(1.0, float(_job.scheduled_time - _job.submit_time + _job.run_time)
                                       /
                                       max(_job.run_time, 10))
                            self.bsld_fcfs_dict[_job.job_id] = (_tmp / self.num_job_in_batch)
                            self.job_queue[j] = Job()
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
        machine_row = int(math.ceil(MAX_MACHINE_SIZE / sq))

        vector = np.zeros(((job_queue_row + machine_row), sq, JOB_FEATURES), dtype=float)
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

        for i in range(MAX_QUEUE_SIZE, MAX_QUEUE_SIZE + MAX_MACHINE_SIZE):
            machine_id = i - MAX_QUEUE_SIZE
            cpu_avail = 1.0
            mem_avail = 1.0
            io_avail = 1.0
            if self.cluster.all_nodes[machine_id].is_free:
                cpu_avail = 1.0
            else:
                running_job_id = self.cluster.all_nodes[machine_id].running_job_id
                running_job = None
                for _j in self.running_jobs:
                    if _j.job_id == running_job_id:
                        running_job = _j
                        break

                reminder = running_job.scheduled_time + running_job.run_time - self.current_timestamp
                cpu_avail = max(MAX_RUN_TIME - reminder, 0) / MAX_RUN_TIME

            vector[int(i / sq), int(i % sq)] = [cpu_avail, mem_avail, io_avail]

        return np.reshape(vector, [-1, (MAX_QUEUE_SIZE + MAX_MACHINE_SIZE) * JOB_FEATURES])

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
            #print ("action:", action)
            #[print(j.job_id, end=' ') for j in self.job_queue]
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
        else:
            # if there is no enough resource for current job, try to backfill using other jobs

            # calculate the expected starting time of current job.
            _needed_processors = job_for_scheduling.request_number_of_processors
            if DEBUG:
                print("try to back fill for job", job_for_scheduling)
            _expected_start_time = self.current_timestamp
            _extra_released_processors = 0

            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            _free_processors_ = self.cluster.free_node * self.cluster.num_procs_per_node
            if DEBUG:
                print("total processors: ", self.cluster.total_node * self.cluster.num_procs_per_node)
                print("free processors, ", _free_processors_)
            for _job in self.running_jobs:
                _free_processors_ += len(_job.allocated_machines) * self.cluster.num_procs_per_node
                released_time = _job.scheduled_time + _job.run_time
                if _free_processors_ >= _needed_processors:
                    _expected_start_time = released_time
                    _extra_released_processors = _free_processors_ - _needed_processors
                    break
            assert _free_processors_ >= _needed_processors

            # find do we have other jobs that do not affect the _expected_start_time
            for j in range(0, MAX_QUEUE_SIZE):
                _schedule_it = False
                _job = self.job_queue[j]
                if _job.job_id == 0 or _job.job_id == job_for_scheduling_index:
                    continue
                assert _job.scheduled_time == -1  # this job should never be scheduled before.
                if not self.cluster.can_allocated(_job):
                    # if this job can not be allocated, skip to next job in the queue
                    continue

                if (_job.run_time + self.current_timestamp) <= _expected_start_time:
                    # if i can finish earlier than the expected start time of job[i], schedule it.
                    _schedule_it = True
                else:
                    # even this _job lasts longer, but it uses less than _extra processors, it does not affect current job too
                    if _job.request_number_of_processors < _extra_released_processors:
                        _schedule_it = True
                        # but, we have to update the _extra_released_processors if we took it
                        _extra_released_processors -= _job.request_number_of_processors

                if _schedule_it:
                    if DEBUG:
                        print("take backfilling: ", _job, " for job, ", job_for_scheduling)
                    _job.scheduled_time = self.current_timestamp
                    _job.allocated_machines = self.cluster.allocate(_job.job_id, _job.request_number_of_processors)
                    self.running_jobs.append(_job)
                    self.scheduled_logs.append(_job)
                    _tmp = max(1.0, float(_job.scheduled_time - _job.submit_time + _job.run_time)
                               /
                               max(_job.run_time, 10))
                    self.scheduled_bsld[_job.job_id] = (_tmp / self.num_job_in_batch)
                    self.job_queue[j] = Job()

        # move time forward
        if DEBUG:
            print("schedule job?", job_for_scheduling, "move time forward, get_this_job_scheduled: ",
                  get_this_job_scheduled,
                  " job queue is empty?: ", self._is_job_queue_empty(),
                  " running jobs?: ", len(self.running_jobs))

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
            else:
                if not self.running_jobs:
                    break
                self.current_timestamp = next_resource_release_time
                self.cluster.release(next_resource_release_machines)
                removed_job = self.running_jobs.pop(0)  # remove the first running job.
                if DEBUG:
                    print("In step, release a job, ", removed_job, " generated free nodes: ", self.cluster.free_node)

        if DEBUG:
            running_queue_machine_number = 0
            for _job in self.running_jobs:
                running_queue_machine_number += len(_job.allocated_machines) * self.cluster.num_procs_per_node
            total = running_queue_machine_number + self.cluster.free_node * self.cluster.num_procs_per_node

            print("Running jobs take ", running_queue_machine_number,
                  " Remaining free processors ", self.cluster.free_node * self.cluster.num_procs_per_node,
                  " total ", total)

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
            if mine < 0.9 * fcfs:
                return [obs, 1000, True, None]
            elif mine < fcfs:
                return [obs, 100, True, None]
            elif mine < 1.1 * fcfs:
                return [obs, 1, True, None]
            else:
                return [obs, -1, True, None]
        else:
            return [obs, 0, False, None]


def heuristic(env, s):
    while True:
        dist = [np.random.uniform(0,1) for _ in range(MAX_QUEUE_SIZE)]
        max_idx = np.argmax(dist)
        if max_idx == 0:
            print ("Id:", max_idx)
        return max_idx

        #job_queue = s[0][0:MAX_QUEUE_SIZE * JOB_FEATURES]
        #job_queue = np.reshape(job_queue, [MAX_QUEUE_SIZE, JOB_FEATURES])
        #if all(a == 0 for a in job_queue[max_idx]):
        #    continue
        #else:
        #    return max_idx


def demo_scheduler(env):
    total_reward = 0.0
    steps = 0
    s = env.reset()
    while True:
        a = heuristic(env, s)
        s, r, done, _ = env.step(a)
        total_reward += r

        if done:
            #print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.5f}".format(steps, total_reward))

        steps += 1
        if done:
            break
    return total_reward


if __name__ == '__main__':
    print(os.getcwd())
    env = HpcEnvJobLegal()
    env.my_init(workload_file='../../../data/lublin_256.swf')
    # env.my_init(workload_file='../../../data/RICC-2010-2.swf')
    '''
    for i in range(0, 5):
        ts, _ = env.get_metrics_using_algorithm(i, 12897, 13532)
        print("algorithm ", i, " execute: ", ts)
    '''
    for i in range(0, 10000):
        demo_scheduler(env)