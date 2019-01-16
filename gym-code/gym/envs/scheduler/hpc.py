import numpy as np
import math
import gym
import sys
import os
from gym.utils import seeding

from gym.envs.scheduler.job import Job
from gym.envs.scheduler.job import Workloads
from gym.envs.scheduler.cluster import Cluster
from gym.envs.scheduler.cluster import Machine


max_queue_size = 50
empty_job_str = "0        0      0    0   0     0    0   0  0 0  0   0   0  0  0 0 0 0"

max_wait_time = 24 * 60 * 60 # assume maximal wait time is 24 hours.
max_run_time = 12 * 60 * 60 # assume maximal runtime is 12 hours

# each job has features
# submit_time, request_number_of_processors, request_time,
# user_id, group_id, executable_number, queue_number
job_feature_size = 3

np.random.seed(1)

# HPC Batch Scheduler Simulation.
# 
# Created by Dong Dai. Licensed on the same terms as the rest of OpenAI Gym.

class HpcEnv(gym.Env):
    def __init__(self, workload_file = ''):
        # workload_file = "./data/RICC-2010-2.swf"
        print ("current working dir: ", os.getcwd())
        print ("loading workloads from dataset:", workload_file)
        super(HpcEnv, self).__init__()
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

        self.cluster = Cluster("Ricc", self.load.max_nodes, self.load.max_procs/self.load.max_nodes)

        self.n_actions = max_queue_size
        self.n_features = max_queue_size * job_feature_size

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        return 0

    def reset(self):
        self.cluster.reset()
        self.random_start()
        return 0





    def build_queue_vector(self):
        vector = np.array([])

        for job in self.job_queue:

            if job.job_id == 0: # empty job
                for i in range(job_feature_size):
                    vector = np.append(vector, [0])
                continue

            submit_time = job.submit_time
            request_processors = job.request_number_of_processors
            request_node = int(math.ceil(float(request_processors) / float(self.cluster.num_procs_per_node)))

            request_time = job.request_time

            # not used for now
            user_id = job.user_id
            group_id = job.group_id
            executable_number = job.executable_number
            queue_number = job.queue_number

            wait_time = self.current_timestamp - submit_time
            normalized_wait_time = min(1, float(wait_time) / float(max_wait_time))
            normalized_request_time = min(1, float(request_time) / float(max_run_time))

            procs_left = self.cluster.free_node - request_node
            if procs_left < 0:
                normalized_proc_left = -1
            else:
                normalized_proc_left = float(procs_left) / float(self.cluster.total_node)

            job_vector = np.array([normalized_wait_time, normalized_proc_left, normalized_request_time])
            vector = np.append(vector, job_vector)

        return vector

    def is_all_jobs_scheduled(self):
        index = self.start
        num = self.size
        for i in range(index, index + num):
            if self.load[i].scheduled_time == 0:
                return False
        return True

    def is_job_queue_empty(self):
        return all(v.job_id == 0 for v in self.job_queue)

    def random_start(self):
        for i in range(max_queue_size):
            self.job_queue[i] = Job(empty_job_str)

        self.start = np.random.randint(0, self.load.size() - max_queue_size*2)
        self.current_point = self.start
        #DOUBT
        self.size = np.random.randint(max_queue_size, max_queue_size * 2)

        self.current_timestamp = self.load[self.start].submit_time
        self.start_timestamp = self.current_timestamp

        """Generate Workloads: look back to fill the cluster"""
        for i in range(max_queue_size):
            index = self.start - 1 - i
            job = self.load[index]
            if self.cluster.can_allocated(job):
                self.running_jobs.append(job)
                job.scheduled_time = self.current_timestamp
                allocated_machines = self.cluster.allocate(job.job_id, job.request_number_of_processors)
                job.allocated_machines = allocated_machines
            else:
                self.workload_size = (i + 1)
                self.workload_start = index
                break

        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))

        """We need to calculate the worst case for this scheduling batch"""
        longest_delay = 0
        starter = self.running_jobs[-1].scheduled_time + self.running_jobs[-1].run_time
        for i in range(self.start, self.start + self.size):
            starter += self.load[i].run_time
            longest_delay += (starter - self.load[i].submit_time)
        self.longest_run_time = starter
        self.longest_average_delay = float(longest_delay) / float(self.size)
        # print "longest run time: ", self.longest_run_time, " longest average delay: ", self.longest_average_delay

    # there are two places we need to consider scheduling: 1) new job submitted; 2) running job finishes
    def forward(self):

        # get the first finish time of all running jobs
        more_jobs_running = False
        first_to_release_time = sys.maxsize
        first_to_release_job = None
        for running_job in self.running_jobs:
            job_finish_time = running_job.scheduled_time + running_job.run_time
            if job_finish_time < first_to_release_time:
                first_to_release_time = job_finish_time
                first_to_release_job = running_job
                more_jobs_running = True

        more_jobs_to_schedule = True
        if self.current_point >= (self.start + self.size): # no job to submit now
            first_to_submit = sys.maxsize
            more_jobs_to_schedule = False
        else:
            first_to_submit = self.load[self.current_point].submit_time

        if more_jobs_to_schedule and more_jobs_running:
            if first_to_submit <= first_to_release_time:
                for i in range(0, len(self.job_queue)):
                    if self.job_queue[i].job_id == 0:
                        self.job_queue[i] = self.load[self.current_point]
                        break

                self.current_point += 1
                self.current_timestamp = first_to_submit
            else:
                self.running_jobs.remove(first_to_release_job)
                self.cluster.release(first_to_release_job.allocated_machines)
                self.current_timestamp = first_to_release_time

        elif more_jobs_running:
            self.running_jobs.remove(first_to_release_job)
            self.cluster.release(first_to_release_job.allocated_machines)
            self.current_timestamp = first_to_release_time

        elif more_jobs_to_schedule:
            for i in range(0, len(self.job_queue)):
                if self.job_queue[i].job_id == 0:
                    self.job_queue[i] = self.load[self.current_point]
                    break

            self.current_point += 1
            self.current_timestamp = first_to_submit

        elif self.is_job_queue_empty():
            return True, np.array([])

        vector = self.build_queue_vector()
        return False, vector

    # observation_, reward, done
    def schedule(self, action):
        scheduled_job = self.job_queue[action]

        if scheduled_job.job_id == 0:                   # scheduling an empty job
            reward = -2.0
            s_ = self.build_queue_vector()
            return s_, reward, False

        # scheduling a too large job
        if not self.cluster.can_allocated(scheduled_job):
            reward = -4.0
            s_ = self.build_queue_vector()
            return s_, reward, False

        scheduled_job.scheduled_time = self.current_timestamp
        allocated_machines = self.cluster.allocate(scheduled_job.job_id, scheduled_job.request_number_of_processors)
        scheduled_job.allocated_machines = allocated_machines
        self.running_jobs.append(scheduled_job)

        self.job_queue[action] = Job(empty_job_str)  # change it to an empty job
        s_ = self.build_queue_vector()               # build the new job vector

        reward = float(0.0)
        # reward function
        if self.is_job_queue_empty() and self.is_all_jobs_scheduled():         # finish scheduling, calculate rewards
            reward = -1.0

            # calculate delay
            total_delay = 0
            average_delay = 0.0
            maximal_delay = 0

            index = self.start
            num = self.size
            for i in range(index, index + num):
                delay = self.load[i].scheduled_time - self.load[i].submit_time
                if delay > maximal_delay:
                    maximal_delay = delay

                total_delay += delay
            average_delay = float(total_delay) / float(num)

            # calculate system utilization
            maximal_runtime = 0
            consumed_node_hours = 0
            for i in range(self.workload_start, self.workload_start + self.workload_size):
                request_nodes = int(
                    math.ceil(
                        float(self.load[i].request_number_of_processors) / float(self.cluster.num_procs_per_node)))
                consumed_node_hours += self.load[i].run_time * request_nodes

            for i in range(index, index + num):
                finish = self.load[i].scheduled_time + self.load[i].run_time

                request_nodes = int(
                    math.ceil(float(self.load[i].request_number_of_processors) / float(self.cluster.num_procs_per_node)))

                consumed_node_hours += self.load[i].run_time * request_nodes

                if finish > maximal_runtime:
                    maximal_runtime = finish
            total_node_hours = (maximal_runtime - self.start_timestamp) * self.cluster.total_node
            system_utilization = float(consumed_node_hours) / float(total_node_hours)

            # we expect larger rewards
            reward += float(self.longest_average_delay - average_delay)/float(self.longest_average_delay) + \
                      float(self.longest_run_time - maximal_runtime) / float(self.longest_run_time)
            print ("reward: ", reward)
            return s_, reward, True

        else:                                           # just -1 because we pay for each scheduling
            reward = -1.0
            return s_, reward, False
    
    
if __name__ == "__main__":
    
    o = HpcEnv("/home/aaroh/Desktop/UNCC_SPRING'19/Dong Dai_IS /deep-batch-scheduler/code_to_use_for_workload_loading/data/RICC-2010-2.swf")
    print(o.seed(),o.np_random.randint(1, o.load.size()))
