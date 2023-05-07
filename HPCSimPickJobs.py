from job import Job, Workloads
from cluster import Cluster
import scipy.signal
import numpy as np
import random
import sys
import os
from gym.spaces import Box, Discrete
from gym.utils import seeding
from gym import Env
import argparse

MAX_QUEUE_SIZE = 128
MLP_SIZE = 256

MAX_WAIT_TIME = 12 * 60 * 60 # assume maximal wait time is 12 hours.
MAX_RUN_TIME = 12 * 60 * 60 # assume maximal runtime is 12 hours

# each job has three features: wait_time, requested_node, runtime, machine states,
JOB_FEATURES = 9
DEBUG = False

JOB_SEQUENCE_SIZE = 256
SKIP_TIME = 360 # skip 60 seconds

class HPCEnv(Env):
    def __init__(self,shuffle=False, backfil=False, skip=False, job_score_type=0, batch_job_slice=0, build_sjf=False): 
        super(HPCEnv, self).__init__()
        print("Initialize Simple HPC Env")

        self.action_space = Discrete(MAX_QUEUE_SIZE)
        self.observation_space = Box(low=0.0, high=1.0,
                                            shape=((JOB_FEATURES) * MAX_QUEUE_SIZE,),
                                            dtype=np.float32)

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.start_idx_last_reset = 0

        self.loads = None
        self.cluster = None

        self.bsld_algo_dict = {}
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_job = False
        self.scheduled_scores = []
        self.enable_preworkloads = False
        self.pre_workloads = []

        self.shuffle = shuffle
        self.backfil = backfil
        self.skip = skip
        # 0: Average bounded slowdown, 1: Average waiting time
        # 2: Average turnaround time, 3: Resource utilization
        self.job_score_type = job_score_type
        self.batch_job_slice = batch_job_slice

        self.build_sjf = build_sjf
        self.sjf_scores = []

    #@profile
    def my_init(self, workload_file = '', sched_file = ''):
        print ("loading workloads from dataset:", workload_file)
        self.loads = Workloads(workload_file)
        self.cluster = Cluster("Cluster", self.loads.max_nodes, self.loads.max_procs/self.loads.max_nodes)
        self.penalty_job_score = JOB_SEQUENCE_SIZE * self.loads.max_exec_time / 10

        if self.build_sjf: #this is for trajectory filtering.
            #calculate SJF scores for all sample sequence and save them here
            index = 0
            if self.batch_job_slice == 0:
                max_index = self.loads.size() - JOB_SEQUENCE_SIZE - 1
            else:
                max_index = min(self.batch_job_slice, self.loads.size()) - JOB_SEQUENCE_SIZE - 1
            print("max index... initializing SJF Score Array", max_index)

            while index <= max_index:
                index += 1
                if index % 100 == 0:
                    print("index", index)

                self.cluster.reset()
                self.loads.reset()

                self.job_queue = []
                self.running_jobs = []
                self.visible_jobs = []
                self.pairs = []

                self.current_timestamp = 0
                self.start = 0
                self.next_arriving_job_idx = 0
                self.last_job_in_batch = 0
                self.num_job_in_batch = 0
                self.scheduled_rl = {}
                self.penalty = 0
                self.pivot_job = False
                self.scheduled_scores = []

                job_sequence_size = JOB_SEQUENCE_SIZE
                self.pre_workloads = []

                self.start = index
                self.start_idx_last_reset = self.start
                self.num_job_in_batch = job_sequence_size
                self.last_job_in_batch = self.start + self.num_job_in_batch
                self.current_timestamp = self.loads[self.start].submit_time
                self.job_queue.append(self.loads[self.start])
                self.next_arriving_job_idx = self.start + 1

                if self.enable_preworkloads:
                    self.gen_preworkloads(job_sequence_size + self.np_random.randint(job_sequence_size))

                self.sjf_scores.append(sum(self.schedule_curr_sequence_reset(self.sjf_score).values()))

            #print(self.sjf_scores)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def f1_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        return (np.log10(request_time if request_time>0 else 0.1) * request_processors + 870 * np.log10(submit_time if submit_time>0 else 0.1))

    def f2_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        # f2: r^(1/2)*n + 25600 * log10(s)
        return (np.sqrt(request_time) * request_processors + 25600 * np.log10(submit_time))

    def f3_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        # f3: r * n + 6860000 * log10(s)
        return (request_time * request_processors + 6860000 * np.log10(submit_time))

    def f4_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        # f4: r * sqrt(n) + 530000 * log10(s)
        return (request_time * np.sqrt(request_processors) + 530000 * np.log10(submit_time))

    def sjf_score(self, job):
        # run_time = job.run_time
        request_time = job.request_time
        submit_time = job.submit_time
        # if request_time is the same, pick whichever submitted earlier 
        return (request_time, submit_time)
    
    def smallest_score(self, job):
        request_processors = job.request_number_of_processors
        submit_time = job.submit_time
        # if request_time is the same, pick whichever submitted earlier 
        return (request_processors, submit_time)

    def wfp_score(self, job):
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        waiting_time = job.scheduled_time-job.submit_time
        return -np.power(float(waiting_time)/request_time, 3)*request_processors

    def uni_score(self,job):
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        waiting_time = job.scheduled_time-job.submit_time

        return -(waiting_time+1e-15)/(np.log2(request_processors+1e-15)*request_time)

    def fcfs_score(self, job):
        submit_time = job.submit_time
        return submit_time

    def gen_preworkloads(self, size):
        # Generate some running jobs to randomly fill the cluster.
        # size = self.np_random.randint(2 * job_sequence_size)
        running_job_size = size
        for i in range(running_job_size):
            _job = self.loads[self.start - i - 1]
            req_num_of_processors = _job.request_number_of_processors
            runtime_of_job = _job.request_time
            job_tmp = Job()
            job_tmp.job_id = (-1 - i)  # to be different from the normal jobs; normal jobs have a job_id >= 0
            job_tmp.request_number_of_processors = req_num_of_processors
            job_tmp.run_time = runtime_of_job
            if self.cluster.can_allocated(job_tmp):
                self.running_jobs.append(job_tmp)
                job_tmp.scheduled_time = max(0, (self.current_timestamp - random.randint(0, max(runtime_of_job, 1))))
                # job_tmp.scheduled_time = max(0, (self.current_timestamp - runtime_of_job/2))
                job_tmp.allocated_machines = self.cluster.allocate(job_tmp.job_id, job_tmp.request_number_of_processors)
                self.pre_workloads.append(job_tmp)
            else:
                break

    def refill_preworkloads(self):
        for _job in self.pre_workloads:
            self.running_jobs.append(_job)
            _job.allocated_machines = self.cluster.allocate(_job.job_id, _job.request_number_of_processors)    

    #@profile
    def reset(self):
        self.cluster.reset()
        self.loads.reset()

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_job = False
        self.scheduled_scores = []

        job_sequence_size = JOB_SEQUENCE_SIZE

        self.pre_workloads = []
        
        assert self.batch_job_slice == 0 or self.batch_job_slice>=job_sequence_size

        if self.build_sjf:
            done = False
            while not done:
                # randomly sample a sequence of jobs from workload (self.start_idx_last_reset + 1) % (self.loads.size() - 2 * job_sequence_size
                if self.batch_job_slice == 0:
                    self.start = self.np_random.randint(job_sequence_size, (self.loads.size() - job_sequence_size - 1))
                else:
                    self.start = self.np_random.randint(job_sequence_size, (self.batch_job_slice - job_sequence_size - 1))

                if self.sjf_scores[self.start] > 10 and self.sjf_scores[self.start] < 150:
                    done = True
        else:
            if self.batch_job_slice == 0:
                self.start = self.np_random.randint(job_sequence_size, (self.loads.size() - job_sequence_size - 1))
            else:
                self.start = self.np_random.randint(job_sequence_size, (self.batch_job_slice - job_sequence_size - 1))

        self.start_idx_last_reset = self.start
        self.num_job_in_batch = job_sequence_size
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.next_arriving_job_idx = self.start + 1



        if self.enable_preworkloads:
            self.gen_preworkloads(job_sequence_size + self.np_random.randint(job_sequence_size))

        self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.sjf_score).values()))
        self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.f1_score).values()))     

        return self.build_observation()

    def reset_for_test(self, num, start):
        self.cluster.reset()
        self.loads.reset()

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_job = False
        self.scheduled_scores = []

        job_sequence_size = num
        assert self.batch_job_slice == 0 or self.batch_job_slice>=job_sequence_size
        if self.batch_job_slice == 0:
            self.start = self.np_random.randint(job_sequence_size, (self.loads.size() - job_sequence_size - 1))
        else:
            self.start = self.np_random.randint(job_sequence_size, (self.batch_job_slice - job_sequence_size - 1))

        self.start_idx_last_reset = self.start
        self.num_job_in_batch = job_sequence_size
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.next_arriving_job_idx = self.start + 1
    
    def skip_for_resources_greedy(self, job, scheduled_logs):
        #note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)

        while not self.cluster.can_allocated(job):
            # schedule nothing, just move forward to next timestamp. It should just add a new job or finish a running job
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

            if self.next_arriving_job_idx < self.last_job_in_batch and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.

    #@profile
    def moveforward_for_resources_backfill_greedy(self, job, scheduled_logs):
        #note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)

        earliest_start_time = self.current_timestamp
        # sort all running jobs by estimated finish time
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        free_processors = self.cluster.free_node * self.cluster.num_procs_per_node
        for running_job in self.running_jobs:
            free_processors += len(running_job.allocated_machines) * self.cluster.num_procs_per_node
            earliest_start_time = (running_job.scheduled_time + running_job.request_time)
            if free_processors >= job.request_number_of_processors:
                break

        while not self.cluster.can_allocated(job):

            # try to backfill as many jobs as possible. Use FCFS
            self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))
            job_queue_iter_copy = list(self.job_queue)
            for _j in job_queue_iter_copy:
                if (self.current_timestamp + _j.request_time) < earliest_start_time:
                    if self.cluster.can_allocated(_j):
                        # we should be OK to schedule the job now
                        assert _j.scheduled_time == -1  # this job should never be scheduled before.
                        _j.scheduled_time = self.current_timestamp
                        _j.allocated_machines = self.cluster.allocate(_j.job_id, _j.request_number_of_processors)
                        self.running_jobs.append(_j)
                        score = self.job_score(_j)   # calculated reward
                        scheduled_logs[_j.job_id] = score
                        self.job_queue.remove(_j)  # remove the job from job queue

            # move to the next timestamp
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines
                
            if self.next_arriving_job_idx < self.last_job_in_batch \
            and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job

    def post_process_score(self, scheduled_logs):
        if self.job_score_type == 0:
            # bsld
            for i in scheduled_logs:
                scheduled_logs[i] /= self.num_job_in_batch
        elif self.job_score_type == 1:
            # wait time
            for i in scheduled_logs:
                scheduled_logs[i] /= self.num_job_in_batch
        elif self.job_score_type == 2:
            # turnaround time
            for i in scheduled_logs:
                scheduled_logs[i] /= self.num_job_in_batch
        elif self.job_score_type == 3:
            total_cpu_hour = (self.current_timestamp - self.loads[self.start].submit_time)*self.loads.max_procs
            for i in scheduled_logs:
                scheduled_logs[i] /= total_cpu_hour
        else:
            raise NotImplementedError
    #@profile
    def schedule_curr_sequence_reset(self, score_fn):
        # schedule the sequence of jobs using heuristic algorithm. 
        scheduled_logs = {}
        # f = False
        # if score_fn.__name__ == "sjf_score":
        #     f = True
        #     num_total = 0
        # start_time = time.time()
        while True:
            self.job_queue.sort(key=lambda j: score_fn(j))
            job_for_scheduling = self.job_queue[0]
            # if f:
            #     num_total += 1
            # if selected job needs more resources, skip scheduling and try again after adding new jobs or releasing some resources
            if not self.cluster.can_allocated(job_for_scheduling):
                if self.backfil:
                    self.moveforward_for_resources_backfill_greedy(job_for_scheduling, scheduled_logs)
                else:
                    self.skip_for_resources_greedy(job_for_scheduling, scheduled_logs)
            
            assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
            job_for_scheduling.scheduled_time = self.current_timestamp
            job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                        job_for_scheduling.request_number_of_processors)
            self.running_jobs.append(job_for_scheduling)
            score = self.job_score(job_for_scheduling)  # calculated reward
            scheduled_logs[job_for_scheduling.job_id] = score
            self.job_queue.remove(job_for_scheduling)

            not_empty = self.moveforward_for_job()
            if not not_empty:
                break
        self.post_process_score(scheduled_logs)
        # if f:
        #     print((time.time()-start_time)/num_total, num_total)
        # reset again
        self.cluster.reset()
        self.loads.reset()
        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.next_arriving_job_idx = self.start + 1

        if self.enable_preworkloads:
            self.refill_preworkloads()

        return scheduled_logs

    def build_critic_observation(self):
        vector = np.zeros(JOB_SEQUENCE_SIZE * 3,dtype=float)
        earlist_job = self.loads[self.start_idx_last_reset]
        earlist_submit_time = earlist_job.submit_time
        pairs = []
        for i in range(self.start_idx_last_reset, self.last_job_in_batch+1):
            job = self.loads[i]
            submit_time = job.submit_time - earlist_submit_time
            request_processors = job.request_number_of_processors
            request_time = job.request_time

            normalized_submit_time = min(float(submit_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5)
            normalized_run_time = min(float(request_time) / float(self.loads.max_exec_time), 1.0 - 1e-5)
            normalized_request_nodes = min(float(request_processors) / float(self.loads.max_procs), 1.0 - 1e-5)

            pairs.append([normalized_submit_time, normalized_run_time, normalized_request_nodes])

        for i in range(JOB_SEQUENCE_SIZE):
            vector[i*3:(i+1)*3] = pairs[i]

        return vector

    def build_observation(self):
        vector = np.zeros((MAX_QUEUE_SIZE) * (JOB_FEATURES), dtype=float)
        self.job_queue.sort(key=lambda job: self.fcfs_score(job))
        self.visible_jobs = []
        for i in range(0, MAX_QUEUE_SIZE):
            if i < len(self.job_queue):
                self.visible_jobs.append(self.job_queue[i])
            else:
                break
        self.visible_jobs.sort(key=lambda j: self.fcfs_score(j))
        if self.shuffle:
            random.shuffle(self.visible_jobs)


        #@ddai: optimize the observable jobs
        self.visible_jobs = []
        if len(self.job_queue) <= MAX_QUEUE_SIZE:
            for i in range(0, len(self.job_queue)):
                self.visible_jobs.append(self.job_queue[i])
        else:
            visible_f1 = []
            f1_index = 0
            self.job_queue.sort(key=lambda job: self.f1_score(job))
            for i in range(0, MAX_QUEUE_SIZE):
                visible_f1.append(self.job_queue[i])
            
            visible_f2 = []
            f2_index = 0
            self.job_queue.sort(key=lambda job: self.f2_score(job))
            for i in range(0, MAX_QUEUE_SIZE):
                visible_f2.append(self.job_queue[i])
            
            visible_sjf = []
            sjf_index = 0
            self.job_queue.sort(key=lambda job: self.sjf_score(job))
            for i in range(0, MAX_QUEUE_SIZE):
                visible_sjf.append(self.job_queue[i])

            visible_small = []
            small_index = 0
            self.job_queue.sort(key=lambda job: self.smallest_score(job))
            for i in range(0, MAX_QUEUE_SIZE):
                visible_small.append(self.job_queue[i])

            visible_random = []
            random_index = 0
            shuffled = list(self.job_queue)
            random.shuffle(shuffled)
            for i in range(0, MAX_QUEUE_SIZE):
                visible_random.append(shuffled[i])

            index = 0

            while index < MAX_QUEUE_SIZE:
                f1_index += 1
                f2_index += 1
                sjf_job = visible_sjf[sjf_index]
                sjf_index += 1
                small_job = visible_small[small_index]
                small_index += 1
                random_job = visible_sjf[random_index]
                random_index += 1
                if (not sjf_job in self.visible_jobs) and index < MAX_QUEUE_SIZE:
                    self.visible_jobs.append(sjf_job)
                    index += 1
                if (not small_job in self.visible_jobs) and index < MAX_QUEUE_SIZE:
                    self.visible_jobs.append(small_job)
                    index += 1
                if (not random_job in self.visible_jobs) and index < MAX_QUEUE_SIZE:
                    self.visible_jobs.append(random_job)
                    index += 1

        self.pairs = []
        add_skip = False
        for i in range(0, MAX_QUEUE_SIZE):
            if i < len(self.visible_jobs) and i < (MAX_QUEUE_SIZE ):
                job = self.visible_jobs[i]
                submit_time = job.submit_time
                request_processors = job.request_number_of_processors
                request_time = job.request_time
                # run_time = job.run_time
                wait_time = self.current_timestamp - submit_time

                # make sure that larger value is better.
                normalized_wait_time = min(float(wait_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5)
                normalized_run_time = min(float(request_time) / float(self.loads.max_exec_time), 1.0 - 1e-5)
                normalized_request_nodes = min(float(request_processors) / float(self.loads.max_procs),  1.0 - 1e-5)

                # add extra parameters, include "Requested Memory", "User Id", "Groupd Id", "Exectuable Id", if its value does not exist in the trace (-1), we set it to 1 by default.
                if job.request_memory == -1:
                    normalized_request_memory = 1
                else:
                    normalized_request_memory = min(float(job.request_memory)/float(self.loads.max_requested_memory), 1.0 - 1e-5)

                if job.user_id == -1:
                    normalized_user_id = 1
                else:
                    normalized_user_id = min(float(job.user_id)/float(self.loads.max_user_id), 1.0-1e-5)

                if job.group_id == -1:
                    normalized_group_id = 1
                else:
                    normalized_group_id = min(float(job.group_id)/float(self.loads.max_group_id), 1.0-1e-5)

                if job.executable_number == -1:
                    normalized_executable_id = 1
                else:
                    normalized_executable_id = min(float(job.executable_number)/float(self.loads.max_executable_number), 1.0-1e-5)

                if self.cluster.can_allocated(job):
                    can_schedule_now = 1.0 - 1e-5
                else:
                    can_schedule_now = 1e-5
                self.pairs.append([job,normalized_wait_time, normalized_run_time, normalized_request_nodes, normalized_request_memory, normalized_user_id, normalized_group_id, normalized_executable_id, can_schedule_now, 1])

            elif self.skip and not add_skip:  # the next job is skip
                add_skip = True
                if self.pivot_job:
                    self.pairs.append([None, 1, 1, 1, 1, 1, 1, 1, 1, 0])
                else:
                    self.pairs.append([None, 1, 1, 1, 1, 1, 1, 1, 0, 0])
            else:
                self.pairs.append([None,0,1,1,1,1,1,1,0, 0])

        for i in range(0, MAX_QUEUE_SIZE):
                
                vector[i*(JOB_FEATURES):(i+1)*(JOB_FEATURES)] = self.pairs[i][1:]

        return np.array(vector, dtype=np.float32)

    #@profile
    def moveforward_for_resources_backfill(self, job):
        #note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)

        earliest_start_time = self.current_timestamp
        # sort all running jobs by estimated finish time
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        free_processors = self.cluster.free_node * self.cluster.num_procs_per_node
        for running_job in self.running_jobs:
            free_processors += len(running_job.allocated_machines) * self.cluster.num_procs_per_node
            earliest_start_time = (running_job.scheduled_time + running_job.request_time)
            if free_processors >= job.request_number_of_processors:
                break

        while not self.cluster.can_allocated(job):
            # try to backfill as many jobs as possible. Use FCFS
            self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))
            job_queue_iter_copy = list(self.job_queue)
            for _j in job_queue_iter_copy:
                if self.cluster.can_allocated(_j) and (self.current_timestamp + _j.request_time) < earliest_start_time:
                    # we should be OK to schedule the job now
                    assert _j.scheduled_time == -1  # this job should never be scheduled before.
                    _j.scheduled_time = self.current_timestamp
                    _j.allocated_machines = self.cluster.allocate(_j.job_id, _j.request_number_of_processors)
                    self.running_jobs.append(_j)
                    score = self.job_score(_j)   # calculated reward
                    self.scheduled_rl[_j.job_id] = score
                    self.job_queue.remove(_j)  # remove the job from job queue

            # move to the next timestamp
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines
                
            if self.next_arriving_job_idx < self.last_job_in_batch \
            and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job
    
    def skip_for_resources(self, job):
        #note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)

        while not self.cluster.can_allocated(job):
            # schedule nothing, just move forward to next timestamp. It should just add a new job or finish a running job
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

            if self.next_arriving_job_idx < self.last_job_in_batch and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.

    #@profile
    def moveforward_for_job(self):
        if self.job_queue:
            return True

        # if we need to add job, but can not add any more, return False indicating the job_queue is for sure empty now.
        if self.next_arriving_job_idx >= self.last_job_in_batch:
            assert not self.job_queue
            return False

        # move forward to add jobs into job queue.
        while not self.job_queue:
            if not self.running_jobs:  # there are no running jobs
                next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
                next_resource_release_machines = []
            else:
                self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
                next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                next_resource_release_machines = self.running_jobs[0].allocated_machines

            if self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
                return True     # job added
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.

    def job_score(self, job_for_scheduling):

        # 0: Average bounded slowdown, 1: Average waiting time
        # 2: Average turnaround time, 3: Resource utilization 4: Average slowdown
        if self.job_score_type == 0:
            # bsld
            _tmp = max(1.0, (float(job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)
                            /
                            max(job_for_scheduling.run_time, 10)))
        elif self.job_score_type == 1:
            #wait time
            _tmp = float(job_for_scheduling.scheduled_time - job_for_scheduling.submit_time)
        elif self.job_score_type == 2:
            # turnaround time
            _tmp = float(job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)
        elif self.job_score_type == 3:
            # utilization
            _tmp = -float(job_for_scheduling.run_time*job_for_scheduling.request_number_of_processors)
        elif self.job_score_type == 4:
            # sld
            _tmp = float(job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)\
                /job_for_scheduling.run_time
        else:
            raise NotImplementedError
        
        return _tmp

    def has_only_one_job(self):
        if len(self.job_queue) == 1:
            return True
        else:
            return False

    def skip_schedule(self):
        # schedule nothing, just move forward to next timestamp. It should 1) add a new job; 2) finish a running job; 3) reach skip time
        next_time_after_skip = self.current_timestamp + SKIP_TIME

        next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
        next_resource_release_machines = []
        if self.running_jobs:  # there are running jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

        if self.next_arriving_job_idx >= self.last_job_in_batch and not self.running_jobs:
            if not self.pivot_job:
                self.pivot_job = True
                return False, 0
            else:
                return False, 0

        if next_time_after_skip < min(self.loads[self.next_arriving_job_idx].submit_time, next_resource_release_time):
            self.current_timestamp = next_time_after_skip
            return False, 0
        
        if self.next_arriving_job_idx < self.last_job_in_batch and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
            self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
            self.job_queue.append(self.loads[self.next_arriving_job_idx])
            self.next_arriving_job_idx += 1
        else:
            self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
            self.cluster.release(next_resource_release_machines)
            self.running_jobs.pop(0)  # remove the first running job.
        return False, 0

    def schedule(self, job_for_scheduling):
        # make sure we move forward and release needed resources
        if not self.cluster.can_allocated(job_for_scheduling):
            if self.backfil:
                self.moveforward_for_resources_backfill(job_for_scheduling)
            else:
                self.skip_for_resources(job_for_scheduling)
        
        # we should be OK to schedule the job now
        assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
        job_for_scheduling.scheduled_time = self.current_timestamp
        job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id, job_for_scheduling.request_number_of_processors)
        self.running_jobs.append(job_for_scheduling)
        score = self.job_score(job_for_scheduling)   # calculated reward
        self.scheduled_rl[job_for_scheduling.job_id] = score
        self.job_queue.remove(job_for_scheduling)  # remove the job from job queue

        # after scheduling, check if job queue is empty, try to add jobs. 
        not_empty = self.moveforward_for_job()
        #print(self.job_queue)

        if not_empty:
            # job_queue is not empty
            return False
        else:
            # job_queue is empty and can not add new jobs as we reach the end of the sequence
            return True

    def valid(self, a):
        action = a[0]
        return self.pairs[action][0]

    #@profile
    def step(self, a):
        #print(f'index: {a} job queue: {len(self.job_queue)}')
        job_for_scheduling = self.pairs[a][0]
        if not job_for_scheduling:
            done, _ = self.skip_schedule()
        else:
            job_for_scheduling = self.pairs[a][0]
            done = self.schedule(job_for_scheduling)          

        if not done:
            obs = self.build_observation()
            return [obs, 0, False, {'best_total': 0, 'rl_total': 0, 'sjf': 0, 'f1': 0, 'rwd2': 0}]
        else:
            self.post_process_score(self.scheduled_rl)
            rl_total = sum(self.scheduled_rl.values())
            best_total = min(self.scheduled_scores)
            sjf = self.scheduled_scores[0]
            f1 = self.scheduled_scores[1]
            rwd2 = (best_total - rl_total)
            rwd = -rl_total
            return [None, rwd, True, {'best_total': best_total, 'rl_total': rl_total, 'sjf': sjf, 'f1': f1, 'rwd2': rwd2}]
    
    def step_for_test(self, a):
        job_for_scheduling = self.pairs[a][0]
        #print(job_for_scheduling)

        if not job_for_scheduling:
            # print("SKIP", end=" ")
            done, _ = self.skip_schedule()
        else:
            job_for_scheduling = self.pairs[a][0]
            done = self.schedule(job_for_scheduling)

        if not done:
            obs = self.build_observation()
            return [obs, 0, False, None]
        else:
            self.post_process_score(self.scheduled_rl)
            rl_total = sum(self.scheduled_rl.values())
            return [None, rl_total, True, None]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default='./data/lublin_256.swf')  # RICC-2010-2
    args = parser.parse_args()
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)

    env = HPCEnv(batch_job_slice=100, build_sjf=True)
    env.seed(0)
    env.my_init(workload_file=workload_file, sched_file=workload_file)
