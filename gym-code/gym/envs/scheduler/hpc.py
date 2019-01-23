import numpy as np
import math
import sys
import os

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.scheduler.job import Job
from gym.envs.scheduler.job import Workloads
from gym.envs.scheduler.cluster import Cluster

# HPC Batch Scheduler Simulation.
# 
# Created by Dong Dai. Licensed on the same terms as the rest of OpenAI Gym.

MAX_QUEUE_SIZE = 50
MAX_WAIT_TIME = 12 * 60 * 60 # assume maximal wait time is 12 hours.
MAX_RUN_TIME = 12 * 60 * 60 # assume maximal runtime is 12 hours
NUM_JOB_SCHEDULERS = 6

# each job has features
# submit_time, request_number_of_processors, request_time,
# user_id, group_id, executable_number, queue_number
JOB_FEATURES = 3
MAX_JOBS_EACH_BATCH = 1000
DEBUG = False


class ResourceRelease:
    def __init__(self, t, job_id, machines, expected_t):
        self.release_time = t
        self.job_id = job_id
        self.release_resources = machines
        self.expected_release_time = expected_t

    def __eq__(self, other):
        return self.job_id == other.job_id


def resource_release_priority(rr):
    return [rr.release_time, rr.job_id]


def expected_resource_release_priority(rr):
    return [rr.expected_release_time, rr.job_id]


class HpcEnv(gym.Env):

    metadata = {
        'render.modes': ['human','ansi']
    }

    def __init__(self, workload_file = ''):
        super(HpcEnv, self).__init__()

        print ("loading workloads from dataset:", workload_file)
        self.loads = Workloads(workload_file)
        self.cluster = Cluster("Ricc", self.loads.max_nodes, self.loads.max_procs/self.loads.max_nodes)

        self.action_space = spaces.Discrete(NUM_JOB_SCHEDULERS)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(JOB_FEATURES * MAX_QUEUE_SIZE + 1,),
                                            dtype=np.float32)

        self.np_random = self.np_random, seed = seeding.np_random(1)

        self.job_queue = []
        for i in range(0, MAX_QUEUE_SIZE):
            self.job_queue.append(Job())

        self.priority_function = None

        self.running_jobs = []
        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.schedule_logs = []
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.next_arriving_job_idx = 0
    
        # slurm
        self.priority_max_age = 0.0
        self.priority_weight_age = 0.0
        self.priority_weight_fair_share = 0.0
        self.priority_favor_small = True
        self.priority_weight_job_size = 0.0
        self.priority_weight_partition = 0.0
        self.priority_weight_qos = 0.0
        self.tres_weight_cpu = 0.0
        self._configure_slurm(1000, 0, 1000, 0, 0, 0, 60 * 60 * 72, True)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the space.

        Two things are done in "reset":
        1. randomly choose a batch or a sample of jobs to learn
        2. randomly generate a bunch of jobs that already been scheduled and running to consume random resources.
        """
        self.cluster.reset()

        self.job_queue = []
        for i in range(0, MAX_QUEUE_SIZE):
            self.job_queue.append(Job())
        self.priority_function = None
        self.running_jobs = []
        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.schedule_logs = []
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.next_arriving_job_idx = 0
        self._configure_slurm(1000, 0, 1000, 0, 0, 0, 60 * 60 * 72, True)

        # randomly choose a start point in current workload
        self.start = self.np_random.randint(self.loads.size() - MAX_JOBS_EACH_BATCH)
        # how many jobs are remainded in the workload
        job_remainder = self.loads.size() - self.start
        # how many jobs in this batch
        self.num_job_in_batch = self.np_random.randint(1, min(job_remainder, MAX_JOBS_EACH_BATCH))
        # the id of the last job in this batch.
        self.last_job_in_batch = self.start + self.num_job_in_batch
        # start scheduling from the first job.
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue[0] = self.loads[self.start]
        # next arriving job would be the second job in the batch
        self.next_arriving_job_idx = self.start + 1

        if DEBUG:
            print ("Reset Env: ", self.loads[self.start], ", ", self.num_job_in_batch, ", ", self.loads[
                self.last_job_in_batch])

        # Generate some running jobs to randomly fill the cluster.
        running_job_size = self.np_random.randint(1, MAX_JOBS_EACH_BATCH)  # size of running jobs.
        for i in range(running_job_size):
            req_num_of_processors = self.np_random.randint(1, self.loads.max_procs) # random number of requests
            runtime_of_job = self.np_random.randint(1, self.loads.max_exec_time)    # random execution time
            job_tmp = Job()
            job_tmp.job_id = (-1 - i) # to be different from the normal jobs; normal jobs have a job_id >= 1
            job_tmp.request_number_of_processors = req_num_of_processors
            job_tmp.run_time = runtime_of_job
            if self.cluster.can_allocated(job_tmp):
                self.running_jobs.append(job_tmp)
                # assume job was randomly generated
                job_tmp.scheduled_time = (self.current_timestamp - self.np_random.randint(runtime_of_job))
                if DEBUG:
                    print ("In reset, allocate for job, ", job_tmp, " with free nodes: ", self.cluster.free_node)
                job_tmp.allocated_machines = self.cluster.allocate(job_tmp.job_id, job_tmp.request_number_of_processors)
            else:
                break

        job_queue_vec = self._build_queue_vector()

        node_utilization = float(self.cluster.used_node)/float(self.cluster.total_node)
        return np.append(job_queue_vec, node_utilization)

    def _build_queue_vector(self):
        vector = np.array([])

        for job in self.job_queue:
            if job.job_id == 0: # empty job
                vector = np.append(vector, np.zeros(JOB_FEATURES))
                continue

            submit_time = job.submit_time
            request_processors = job.request_number_of_processors
            request_time = job.request_time
            run_time = job.run_time
            # not used for now
            #user_id = job.user_id
            #group_id = job.group_id
            #executable_number = job.executable_number
            #queue_number = job.queue_number

            wait_time = self.current_timestamp - submit_time
            normalized_wait_time = min(1.0, float(wait_time) / float(MAX_WAIT_TIME))
            normalized_request_time = min(1.0, float(request_time) / float(MAX_RUN_TIME))
            normalized_run_time = min(1.0, float(run_time) / float(MAX_RUN_TIME))
            normalized_request_nodes = min(1.0, float(request_processors) / float(self.loads.max_procs))

            job_vector = np.array([normalized_wait_time, normalized_run_time, normalized_request_nodes])
            vector = np.append(vector, job_vector)

        return vector

    def _is_job_queue_empty(self):
        return all(v.job_id == 0 for v in self.job_queue)

    def slurm_priority(self, job):
        age_factor = float(job.slurm_age) / float(self.priority_max_age)
        if age_factor > 1:
            age_factor = 1.0
        if self.priority_favor_small:
            size_weight = self.priority_weight_job_size * (1.0 - job.slurm_job_size)
        else:
            size_weight = self.priority_weight_job_size * job.slurm_job_size

        priority_score = \
            self.priority_max_age * age_factor + self.priority_weight_fair_share * job.slurm_fair + size_weight \
            + self.priority_weight_partition * job.slurm_partition + self.priority_weight_qos * job.slurm_qos \
            + self.tres_weight_cpu * job.slurm_tres_cpu

        return priority_score

    def fcfs_priority(self, job):
        return job.submit_time

    def smallest_job_first(self, job):
        return job.number_of_allocated_processors

    def largest_job_first(self, job):
        return sys.maxsize - job.number_of_allocated_processors

    def shortest_job_first(self, job):
        return job.request_time

    def longest_job_first(self, job):
        return sys.maxsize - job.request_time

    def _configure_slurm(self, age, fair_share, job_size, partition, qos, tres_cpu, max_age, favor_small):
        self.priority_weight_age = age
        self.priority_max_age = max_age
        self.priority_weight_fair_share = fair_share
        self.priority_weight_job_size = job_size
        self.priority_favor_small = favor_small
        self.priority_weight_partition = partition
        self.priority_weight_qos = qos
        self.tres_weight_cpu = tres_cpu

    def _is_job_queue_full(self):
        for job in self.job_queue:
            if job.job_id == 0:
                return False
        return True

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # Action: scheduling algorithms
        scheduler_algs = {
            0: self.slurm_priority,
            1: self.fcfs_priority,
            2: self.smallest_job_first,
            3: self.shortest_job_first,
            4: self.largest_job_first,
            5: self.longest_job_first
        }

        # action is one of the defined scheduler. 
        # action should be a scalar, calculated based on Maximam Function on the NN outputs
        self.priority_function = scheduler_algs.get(action, self.fcfs_priority)
        self.job_queue.sort(key=lambda j: (self.priority_function(j)))

        scheduled_jobs_in_step = []

        # try to schedule all jobs in the queue
        for i in range(0, MAX_QUEUE_SIZE):
            check_for_schedule = self.job_queue[i]

            assert self.job_queue[i].job_id >= 0
            if self.job_queue[i].job_id == 0:
                continue
            if self.cluster.can_allocated(self.job_queue[i]):
                self.job_queue[i].scheduled_time = self.current_timestamp
                if DEBUG:
                    print("In step, schedule a job, ", self.job_queue[i], " with free nodes: ", self.cluster.free_node)
                self.job_queue[i].allocated_machines = self.cluster.allocate(self.job_queue[i].job_id, self.job_queue[i].request_number_of_processors)
                self.running_jobs.append(self.job_queue[i])
                self.schedule_logs.append(self.job_queue[i])
                scheduled_jobs_in_step.append(self.job_queue[i])
                self.job_queue[i] = Job()       # remove the job from job queue
                break # @important: each step, we just scheduled one job!!!
            else:
                # if there is no enough resource for current job, try to backfill the jobs behind it
                # calculate the expected starting time of job[i].
                _needed_processors = self.job_queue[i].request_number_of_processors
                _expected_start_time = self.current_timestamp
                _extra_released_processors = 0

                self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
                _released_resources = self.cluster.free_node * self.cluster.num_procs_per_node
                for _job in self.running_jobs:
                    _released_resources += _job.request_number_of_processors
                    released_time = _job.scheduled_time + _job.run_time
                    if _released_resources >= _needed_processors:
                        _expected_start_time = released_time
                        _extra_released_processors = _released_resources - _needed_processors
                        break
                assert _released_resources >= _needed_processors

                # find do we have later jobs that do not affect the _expected_start_time
                for j in range(i + 1, MAX_QUEUE_SIZE):
                    _schedule_it = False
                    _job = self.job_queue[j]
                    if _job.job_id == 0:
                        continue

                    if not self.cluster.can_allocated(_job):
                        # if this job can not be allocated, skip to next job in the queue
                        continue

                    if (_job.run_time + self.current_timestamp) <= _expected_start_time:
                        # if i can finish earlier than the expected start time of job[i], schedule it.
                        _schedule_it = True
                    else:
                        if _job.request_number_of_processors < _extra_released_processors:
                            # if my allocation is small enough, not affecting anything, schedule it
                            _schedule_it = True
                            # but, we have to update the _extra_released_processors if we took it
                            _extra_released_processors -= _job.request_number_of_processors

                    if _schedule_it:
                        if DEBUG:
                            print ("take backfilling: ", _job, " for job, ", check_for_schedule)
                        _job.scheduled_time = self.current_timestamp
                        _job.allocated_machines = self.cluster.allocate(_job.job_id, _job.request_number_of_processors)
                        self.running_jobs.append(_job)
                        self.schedule_logs.append(_job)
                        scheduled_jobs_in_step.append(_job)
                        self.job_queue[i] = Job()
                break

        ts_before_move_forward = self.current_timestamp

        # move to next time step
        if not self.running_jobs: # there are no running jobs
            next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
            next_resource_release_machines = []
        else:
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines
        
        while True:
            if self.next_arriving_job_idx < self.last_job_in_batch \
                    and self.loads[self.next_arriving_job_idx].submit_time < next_resource_release_time \
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
                removed_job = self.running_jobs.pop(0) # remove the first running job.
                if DEBUG:
                    print("In step, release a job, ", removed_job, " generated free nodes: ", self.cluster.free_node)
                break
        
        # calculate reward
        # There are different ways to calculate it. We use the HotNet paper
        # minimizing the average slowdown of all the jobs waiting for service.
        reward = 0.0
        # reward += -1.0 # we want to schedule all the jobs as soon as possible.
        for _job in scheduled_jobs_in_step:
            assert _job.scheduled_time != 0
            _slow_down = 1.0 + _job.scheduled_time - _job.submit_time
            reward += (0 - 1.0 / float(_slow_down))

        job_queue_vec = self._build_queue_vector()
        
        done = True
        for i in range(self.start, self.last_job_in_batch):
            if self.loads[i].scheduled_time == 0:  # have at least one job in the batch who has not been scheduled
                done = False
                break

        node_utilization = float(self.cluster.used_node) / float(self.cluster.total_node)
        return [np.append(job_queue_vec, node_utilization), reward, done, None]


def heuristic(env, s):
    action = np.random.randint(0, 6)
    print ("take action, ", action)
    return action


def demo_scheduler(env):
    env.seed()
    total_reward = 0.0
    steps = 0
    s = env.reset()
    while True:
        a = heuristic(env, s)
        s, r, done, _ = env.step(a)
        total_reward += r

        if steps >= 0 or done:
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.4f}".format(steps, total_reward))

        steps += 1
        if done: 
            break
    return total_reward


if __name__ == '__main__':
    print (os.getcwd())
    demo_scheduler(HpcEnv(workload_file = '../../../data/RICC-2010-2.swf'))