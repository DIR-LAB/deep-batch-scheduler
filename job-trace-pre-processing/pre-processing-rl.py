import sys
import json
import math

from hpc.envs.job import Job, Workloads
from hpc.envs.cluster import Machine, Cluster

MAX_QUEUE_SIZE = 64
MAX_JOBS_EACH_BATCH = 64
NUM_OF_ALGS = 9

class RLProcessor():

    def __init__(self, workload_file, output_file):
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

        self.Metrics_Queue_Length = 0  # Determine the quality of the job sequence.
        self.Metrics_Probe_Times = 0  # Determine the quality of the job sequence.
        self.Metrics_Total_Execution_Time = 0  # Max(job.scheduled_time + job.run_time)
        self.Metrics_Average_Response_Time = 0  # (job.scheduled_time + job.run_time - job.submit_time) / num_of_jobs
        self.Metrics_Average_Slow_Down = 0  # (job.scheduled_time - job.submit_time) / num_of_jobs
        self.Metrics_Average_BSLD = 0.0     # bounded slowdown objective function (see paper SC17)
        self.Metrics_System_Utilization = 0  # (cluster.used_node * t_used / cluster.total_node * t_max)

        self.scheduler_algs = {
                0: self.fcfs_priority,
                1: self.smallest_job_first,
                2: self.shortest_job_first,
                3: self.largest_job_first,
                4: self.longest_job_first,
                5: self.wfp_3,
                6: self.unicef,
                7: self.wfp,
                8: self.fcsj
        }

        print("loading workloads from dataset:", workload_file)
        self.loads = Workloads(workload_file)
        self.cluster = Cluster("Ricc", self.loads.max_nodes, self.loads.max_procs / self.loads.max_nodes)
        self.output_file = output_file

    def wfp(self, job):
        wait_time = self.current_timestamp - job.submit_time
        tmp = float(wait_time) / float(job.run_time + 0.0001)
        return 0 - tmp * job.request_number_of_processors

    def wfp_3(self, job):
        wait_time = self.current_timestamp - job.submit_time
        tmp = float(wait_time) / float(job.run_time + 0.0001)
        return 0 - (tmp ** 3) * job.request_number_of_processors

    def unicef(self, job):
        wait_time = self.current_timestamp - job.submit_time
        if job.job_id == 0:
            return 0

        round = int(math.ceil(job.request_number_of_processors / 8)) * 8
        if round == 0:
            round = 8

        t = (math.log2(round) * (job.run_time))
        if  t == 0:
            print("job:", job, "job request processor", round, "job runtime", job.run_time)
            t = (math.log2(round) * (job.run_time + 0.0001))
        return 0 - (float(wait_time) / t)

    def fcsj(self, job):
        return job.submit_time / (job.run_time + 0.0001)

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

    def pre_process_job_trace(self):
        """ We preprocess all the jobs, eliminating the job sequence that will not change the scheduling decisions.
        Metrics: if no matter which scheduling algorithm is used, most of the time, job queue will only have one
        job, this is really not the case for our agent to get any meaningful information and they shall be eliminated.
        Also, calculate what the existing algorithms can do, so that we do not need to re-calculate again.
        """
        print("in pre_process_job_trace", self.loads.size())
        '''
            json data looks like this:
            {
                i:{          //i is the start index
                0:[metrics], //fcfs
                1:[metrics], //small
                2:[metrics], //short
                3:[metrics], //large
                4:[metrics], //long
                }, 
                ...
            }
            [metrics] = self.Metrics_Total_Execution_Time, self.Metrics_Average_Slow_Down,
                self.Metrics_Average_Response_Time, utilization, average_queue_size
        '''
        out_dict = {}

        with open(self.output_file, 'w') as f:
            for i in range(0, self.loads.size() - MAX_JOBS_EACH_BATCH):
                high_quality = False
                metrics_dict = {}

                for j in range(0, NUM_OF_ALGS):
                    metrics_list, s_log, average_queue_size = \
                        self.get_metrics_using_algorithm(j, i, (i + MAX_JOBS_EACH_BATCH))
                    metrics_list.append(average_queue_size)
                    # metrics_list.append(s_log)

                    metrics_dict[j] = metrics_list
                    # print(j, "-", average_queue_size, end=", ")
                    if average_queue_size > 2:
                        high_quality = True

                if high_quality:
                    out_dict[i] = metrics_dict
                    print("dict size,", len(out_dict))
                print("Process", i, "as high/low quality sequence", high_quality)

            print ("Size of high quality sequences:", len(out_dict))
            json.dump(out_dict, f)

    def get_metrics_using_algorithm(self, algorithm_id, start, end):
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

        self.start = start
        self.last_job_in_batch = end
        self.num_job_in_batch = self.last_job_in_batch - self.start
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue[0] = self.loads[self.start]
        self.next_arriving_job_idx = self.start + 1

        self.priority_function = self.scheduler_algs.get(algorithm_id)

        self.Metrics_Queue_Length = 0
        self.Metrics_Probe_Times = 0
        self.Metrics_Total_Execution_Time = 0
        self.Metrics_Average_Slow_Down = 0
        self.Metrics_Average_BSLD = 0.0
        self.Metrics_Average_Response_Time = 0
        self.Metrics_System_Utilization = 0

        while True:

            self.job_queue.sort(key=lambda j: (self.priority_function(j)))

            scheduled_jobs_in_step = []
            get_this_job_scheduled = False

            # try to schedule all jobs in the queue
            for i in range(0, MAX_QUEUE_SIZE):
                assert self.job_queue[i].job_id >= 0

                if self.job_queue[i].job_id == 0:
                    continue

                if self.cluster.can_allocated(self.job_queue[i]):
                    assert self.job_queue[i].scheduled_time == -1  # this job should never be scheduled before.
                    self.job_queue[i].scheduled_time = self.current_timestamp
                    self.job_queue[i].allocated_machines = self.cluster.allocate(self.job_queue[i].job_id,
                                                                                 self.job_queue[i].request_number_of_processors)
                    self.running_jobs.append(self.job_queue[i])
                    self.schedule_logs.append(self.job_queue[i])
                    scheduled_jobs_in_step.append(self.job_queue[i])
                    get_this_job_scheduled = True
                    self.Metrics_Queue_Length += self._job_queue_size()
                    self.Metrics_Probe_Times += 1
                    self.Metrics_Total_Execution_Time = max(self.Metrics_Total_Execution_Time,
                                                            self.job_queue[i].scheduled_time + self.job_queue[i].run_time)
                    self.Metrics_Average_Slow_Down += (self.job_queue[i].scheduled_time - self.job_queue[i].submit_time)
                    self.Metrics_Average_BSLD += max(1.0, (float(
                        self.job_queue[i].scheduled_time - self.job_queue[i].submit_time + self.job_queue[i].run_time) / max(
                            self.job_queue[i].run_time, 10)))
                    self.Metrics_Average_Response_Time += (self.job_queue[i].scheduled_time
                                                           - self.job_queue[i].submit_time + self.job_queue[i].run_time)
                    self.Metrics_System_Utilization += (self.job_queue[i].run_time *
                                                        self.job_queue[i].request_number_of_processors)
                    self.job_queue[i] = Job()  # remove the job from job queue
                    break
                else:
                    # if there is no enough resource for current job, try to backfill the jobs behind it
                    _needed_processors = self.job_queue[i].request_number_of_processors
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
                    for j in range(i + 1, MAX_QUEUE_SIZE):
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
                            self.Metrics_Queue_Length += self._job_queue_size()
                            self.Metrics_Probe_Times += 1
                            self.Metrics_Total_Execution_Time = max(self.Metrics_Total_Execution_Time,
                                                                    _job.scheduled_time + _job.run_time)
                            self.Metrics_Average_Slow_Down += (_job.scheduled_time - _job.submit_time)
                            self.Metrics_Average_BSLD += max(1.0,
                                                             float(_job.scheduled_time - _job.submit_time + _job.run_time)
                                                             /
                                                             max(_job.run_time, 10))
                            self.Metrics_Average_Response_Time += (_job.submit_time - _job.submit_time + _job.run_time)
                            self.Metrics_System_Utilization += (_job.run_time * _job.request_number_of_processors)
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

        utilization = float(self.Metrics_System_Utilization) / float(self.cluster.num_procs_per_node *
                                                                     self.cluster.total_node *
                                                                     self.Metrics_Total_Execution_Time)
        bsld = self.Metrics_Average_BSLD / MAX_JOBS_EACH_BATCH
        average_queue_size = float(self.Metrics_Queue_Length) / float(self.Metrics_Probe_Times)

        return [self.Metrics_Total_Execution_Time, self.Metrics_Average_Slow_Down, bsld,
                self.Metrics_Average_Response_Time, utilization], self.schedule_logs, average_queue_size

if __name__ == '__main__':
    rlp = RLProcessor(workload_file="../data/RICC-2010-2.swf", output_file="../data/RICC-RL-BSLD-64.txt")
    rlp.pre_process_job_trace()