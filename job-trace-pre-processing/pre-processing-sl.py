import sys
import json

import math
import numpy as np

from hpc.envs.job import Job, Workloads
from hpc.envs.cluster import Machine, Cluster

MAX_QUEUE_SIZE = 64
MAX_MACHINE_SIZE = 1024

MAX_JOBS_EACH_BATCH = 64
JOB_FEATURES = 3

MAX_WAIT_TIME = 12 * 60 * 60 # assume maximal wait time is 12 hours.
MAX_RUN_TIME = 12 * 60 * 60 # assume maximal runtime is 12 hours

class SLProcessor:
    def __init__(self, workload_file):
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

        self.scheduler_algs = {
                0: self.fcfs_priority,
                1: self.smallest_job_first,
                2: self.shortest_job_first,
                3: self.largest_job_first,
                4: self.longest_job_first
        }

        print("loading workloads from dataset:", workload_file)
        self.loads = Workloads(workload_file)
        self.cluster = Cluster("Ricc", self.loads.max_nodes, self.loads.max_procs / self.loads.max_nodes)

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

    def build_observation(self, scheduled_job):
        assert scheduled_job.job_id != 0
        sq = int(math.ceil(math.sqrt(MAX_QUEUE_SIZE)))
        job_queue_row = sq
        machine_row = int(math.ceil(MAX_MACHINE_SIZE / sq))

        vector = np.zeros(((job_queue_row + machine_row), sq, JOB_FEATURES), dtype=float)
        # self.job_queue.sort(key=lambda j: j.request_number_of_processors)

        for i in range(0, MAX_QUEUE_SIZE):
            job = self.job_queue[i]

            submit_time = job.submit_time
            request_processors = job.request_number_of_processors
            request_time = job.request_time
            run_time = job.run_time
            # not used for now
            # user_id = job.user_id
            # group_id = job.group_id
            # executable_number = job.executable_number
            # queue_number = job.queue_number

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

        new_index = 0
        for i in range(0, MAX_QUEUE_SIZE):
            if self.job_queue[i].job_id == scheduled_job.job_id:
                new_index = i
                break
        return np.reshape(vector, [-1, (MAX_QUEUE_SIZE + MAX_MACHINE_SIZE) * JOB_FEATURES]), new_index

    def run_scheduler_to_generate_log(self, algorithm_id, f):
        self.cluster.reset()
        self.loads.reset()

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

        self.start = 0
        self.last_job_in_batch = self.loads.size()
        self.num_job_in_batch = self.last_job_in_batch - self.start
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue[0] = self.loads[self.start]
        self.next_arriving_job_idx = self.start + 1

        self.priority_function = self.scheduler_algs.get(algorithm_id)

        sample_cnt = 0
        sample_json = []

        while True:

            all_jobs = list(self.job_queue)
            all_jobs.sort(key=lambda j: (self.priority_function(j)))

            scheduled_jobs_in_step = []
            get_this_job_scheduled = False

            # try to schedule all jobs in the queue
            for i in range(0, MAX_QUEUE_SIZE):
                assert all_jobs[i].job_id >= 0

                if all_jobs[i].job_id == 0:
                    continue

                job_for_scheduling = None
                job_for_scheduling_index = -1
                for idx in range(0, MAX_QUEUE_SIZE):
                    if self.job_queue[idx].job_id == all_jobs[i].job_id:
                        job_for_scheduling = self.job_queue[idx]
                        job_for_scheduling_index = idx
                        break
                assert job_for_scheduling is not None
                assert job_for_scheduling_index != -1

                if self.cluster.can_allocated(job_for_scheduling):
                    # print ("check job", self.job_queue[i])
                    assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
                    job_for_scheduling.scheduled_time = self.current_timestamp
                    job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                                  job_for_scheduling.request_number_of_processors)
                    self.running_jobs.append(job_for_scheduling)
                    self.schedule_logs.append(job_for_scheduling)
                    scheduled_jobs_in_step.append(job_for_scheduling)
                    get_this_job_scheduled = True
                    # output a training sample
                    ja, new_idx = self.build_observation(job_for_scheduling)
                    l = np.squeeze(ja, axis=0).tolist()
                    sample = {}
                    sample['observe'] = l
                    sample['label'] = new_idx
                    sample_json.append(sample)
                    print(len(sample_json))
                    sample_cnt += 1
                    self.job_queue[new_idx] = Job()  # remove the job from job queue
                    break
                else:
                    # if there is no enough resource for current job, try to backfill the jobs behind it
                    _needed_processors = job_for_scheduling.request_number_of_processors
                    _expected_start_time = self.current_timestamp
                    _extra_released_processors = 0

                    self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
                    _released_resources = self.cluster.free_node * self.cluster.num_procs_per_node
                    for _job in self.running_jobs:
                        _released_resources += len(_job.allocated_machines) * self.cluster.num_procs_per_node
                        released_time = _job.scheduled_time + _job.run_time
                        if _released_resources >= _needed_processors:
                            _expected_start_time = released_time
                            _extra_released_processors = _released_resources - _needed_processors
                            break
                    assert _released_resources >= _needed_processors

                    # find do we have later jobs that do not affect the _expected_start_time
                    for j in range(0, MAX_QUEUE_SIZE):
                        _schedule_it = False
                        _job = self.job_queue[j]
                        if _job.job_id == 0:
                            continue

                        assert _job.scheduled_time == -1  # this job should never be scheduled before.

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
                            _job.scheduled_time = self.current_timestamp
                            _job.allocated_machines = self.cluster.allocate(_job.job_id,
                                                                            _job.request_number_of_processors)
                            self.running_jobs.append(_job)
                            self.schedule_logs.append(_job)
                            scheduled_jobs_in_step.append(_job)
                            self.job_queue[j] = Job()
                    break

            while not get_this_job_scheduled or self._is_job_queue_empty():
                # when the job queue is empty and there is no running job. we just add more jobs into the queue.
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
                            # current timestamp may be larger than next_arriving_job's submit time because
                            # job queue was full and we move forward to release resources.
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

            # @update: we do not give reward until we finish scheduling everything.
            if done:
                break

        json.dump(sample_json, f)

if __name__ == '__main__':
    slp = SLProcessor(workload_file="../data/RICC-2010-2.swf")
    with open("../data/RICC-SL-Shortest.txt", 'w') as f:
        slp.run_scheduler_to_generate_log(2, f)