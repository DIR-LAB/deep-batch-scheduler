from job import Workloads
from cluster import Cluster

import numpy as np
import sys
import math
import heapq

np.random.seed(1)


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


class Scheduler:
    def __init__(self, scheduler_type):
        self.job_queue = []
        self.type = scheduler_type
        self.priority_function = self.random_priority

        self.supported_priority = {
            "slurm": self.slurm_priority,
            "fcfs": self.fcfs_priority,
            "small": self.smallest_job_first,
            "short": self.shortest_job_first,
            "random": self.random_priority
        }

        self.busy_cpu_hours = 0
        self.finish_time = 0
        self.idle_cluster_hours = 0
        self.job_wait_time_total = 0
        self.job_wait_time_max = 0
        self.job_total = 0

        self.scheduling_system_status = []
        self.scheduling_decisions = []

        self.backfilling = True
        self.backfilling_first_fit = True
        self.backfilling_time = 0

        self.priority_max_age = 0.0
        self.priority_weight_age = 0.0
        self.priority_weight_fair_share = 0.0
        self.priority_favor_small = True
        self.priority_weight_job_size = 0.0
        self.priority_weight_partition = 0.0
        self.priority_weight_qos = 0.0
        self.tres_weight_cpu = 0.0

        self.logging = False

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

    def shortest_job_first(self, job):
        return job.request_time

    def random_priority(self, job):
        return job.random_id

    def configure_slurm(self, age, fair_share, job_size, partition, qos, tres_cpu, max_age, favor_small):
        self.priority_weight_age = age
        self.priority_max_age = max_age
        self.priority_weight_fair_share = fair_share
        self.priority_weight_job_size = job_size
        self.priority_favor_small = favor_small
        self.priority_weight_partition = partition
        self.priority_weight_qos = qos
        self.tres_weight_cpu = tres_cpu

    def schedule(self, cluster, workloads):
        self.job_total = workloads.size()
        # self.job_total = 1000
        all_jobs = workloads.all_jobs
        job_index = 0
        current_ts = 0

        resource_release_ts = []
        resource_release_expected = []

        self.priority_function = self.supported_priority.get(self.type, self.random_priority)

        while True:

            if self.logging:
                if job_index < self.job_total:
                    print ("Status Report", "current time:", current_ts, "job Id:", all_jobs[job_index].job_id,
                    "job submit time", all_jobs[job_index].submit_time + "resource release", len(resource_release_ts),
                    "next resource release time", "null" if len(resource_release_ts) == 0 else resource_release_ts[
                        0].release_time,
                    "queue size", len(self.job_queue))

            # actual resource releasing
            while resource_release_ts:
                if resource_release_ts[0][1].release_time <= current_ts:
                    [_, release] = heapq.heappop(resource_release_ts)
                    cluster.release(release.release_resources)

                    for [p, rre] in resource_release_expected:
                        if rre == release:
                            [move_p, move_rr] = [p, rre]
                            break
                    resource_release_expected.remove([move_p, move_rr])
                else:
                    break

            # submit all pending jobs
            while job_index < self.job_total and all_jobs[job_index].submit_time <= current_ts:
                job = all_jobs[job_index]
                heapq.heappush(self.job_queue, [self.priority_function(job), job])
                job.request_number_of_nodes = \
                    int(math.ceil(float(job.request_number_of_processors)/float(cluster.num_procs_per_node)))
                self.busy_cpu_hours += job.number_of_allocated_processors * job.run_time
                job.slurm_in_queue_time = current_ts
                job.slurm_job_size = \
                    float(job.number_of_allocated_processors) / float(cluster.total_node * cluster.num_procs_per_node)
                job_index += 1
                if job_index % 5000 == 0:
                    print ("Scheduling job:", job_index)

            # schedule jobs
            if self.type == "slurm" or self.type == "random":
                tmp_job_queue = []
                for [_, j] in self.job_queue:
                    if self.type == "slurm":
                        j.slurm_age = current_ts - j.slurm_in_queue_time
                    if self.type == "random":
                        if np.random.randint(10) % 2 == 0:
                            j.random_id = j.submit_time
                        else:
                            j.random_id = j.request_time
                    heapq.heappush(tmp_job_queue, [self.priority_function(j), j])

                # rebuild priority queue
                self.job_queue = tmp_job_queue

            while self.job_queue:
                [_, job] = self.job_queue[0]
                _free_nodes_status = cluster.free_node
                machines = cluster.allocate(job.job_id, job.number_of_allocated_processors)
                if machines:
                    _current_job_queue = list(self.job_queue)
                    [_, job] = heapq.heappop(self.job_queue)
                    rr = ResourceRelease(job.run_time + current_ts, job.job_id, machines, job.request_time + current_ts)
                    heapq.heappush(resource_release_ts, [resource_release_priority(rr), rr])
                    heapq.heappush(resource_release_expected, [expected_resource_release_priority(rr), rr])

                    _current_system_status = {"job_queue": _current_job_queue,
                                              "free_nodes": _free_nodes_status,
                                              "current_ts": current_ts}
                    self.scheduling_system_status.append(_current_system_status)
                    self.scheduling_decisions.append(job)

                    self.job_wait_time_total += (current_ts - job.submit_time)
                    if (current_ts - job.submit_time) > self.job_wait_time_max:
                        self.job_wait_time_max = current_ts - job.submit_time
                    job.scheduled_time = current_ts

                else:
                    if self.backfilling:
                        free_nodes = cluster.free_node
                        request_processors = job.number_of_allocated_processors
                        request_nodes = int(math.ceil(float(request_processors) / float(cluster.num_procs_per_node)))
                        expected_start_time = current_ts
                        for [_, err] in resource_release_expected:
                            free_nodes += len(err.release_resources)
                            if free_nodes >= request_nodes:
                                expected_start_time = err.expected_release_time
                                break

                        while True:
                            window = 0
                            backfilling_p = 0
                            backfilling_job = None
                            _free_nodes = cluster.free_node

                            for [p, job] in self.job_queue:
                                if job.request_number_of_nodes == -1:
                                    job.request_number_of_nodes = \
                                        int(math.ceil(float(job.request_number_of_processors)/float(cluster.num_procs_per_node)))

                                if job.request_number_of_nodes <= _free_nodes \
                                        and (job.request_time + current_ts) < expected_start_time:

                                    if self.backfilling_first_fit:
                                        backfilling_p = p
                                        backfilling_job = job
                                        break
                                    else:
                                        if job.request_time * job.number_of_allocated_processors > window:
                                            window = job.request_time * job.number_of_allocated_processors
                                            backfilling_p = p
                                            backfilling_job = job

                            if backfilling_job is not None:

                                _current_job_queue = list(self.job_queue)
                                _free_nodes_status = cluster.free_node

                                self.backfilling_time += 1
                                self.job_queue.remove([backfilling_p, backfilling_job])
                                filled_machine = cluster.allocate(backfilling_job.job_id,
                                                                  backfilling_job.number_of_allocated_processors)
                                if not filled_machine:
                                    print ("should not happen!")
                                rr = ResourceRelease(backfilling_job.run_time + current_ts,
                                                     backfilling_job.job_id, filled_machine,
                                                     backfilling_job.request_time + current_ts)
                                heapq.heappush(resource_release_ts, [resource_release_priority(rr), rr])
                                heapq.heappush(resource_release_expected, [expected_resource_release_priority(rr), rr])

                                _current_system_status = {"job_queue": _current_job_queue,
                                                          "free_nodes": _free_nodes_status,
                                                          "current_ts": current_ts}
                                self.scheduling_system_status.append(_current_system_status)
                                self.scheduling_decisions.append(backfilling_job)

                                self.job_wait_time_total += (current_ts - backfilling_job.submit_time)
                                if (current_ts - backfilling_job.submit_time) > self.job_wait_time_max:
                                    self.job_wait_time_max = current_ts - backfilling_job.submit_time

                                backfilling_job.scheduled_time = current_ts
                            else:
                                break

                    if self.logging:
                        print ("no resource for job", job.job_id)

                    break

            # time travel
            old_ts = current_ts
            if job_index < self.job_total:
                if not resource_release_ts:
                    time = sys.maxsize
                else:
                    time = resource_release_ts[0][1].release_time

                current_ts = min(all_jobs[job_index].submit_time, time)

                if len(self.job_queue) == 0 and cluster.is_idle():
                    self.idle_cluster_hours += (current_ts - old_ts)

                continue

            if self.job_queue:
                if resource_release_ts:
                    current_ts = resource_release_ts[0][1].release_time
                else:
                    print ("Job queue is not empty, but no resources will be released... Error")
            else:
                if resource_release_ts:
                    current_ts = resource_release_ts[0][1].release_time
                else:
                    self.finish_time = current_ts
                    print ("Finish scheduling all jobs...")
                    break

    def print_scheduling_results(self, cluster):
        cluster_utilization = float(self.busy_cpu_hours) * 100 / (
                float(self.finish_time - self.idle_cluster_hours) * cluster.total_node * cluster.num_procs_per_node)
        average_wait_time = float(self.job_wait_time_total) / float(self.job_total)
        s = "Finish scheduling at: {:>5} | cluster utilization: {:>5}% | average job wait time: {:>5} | maximal job " \
            "wait time: {:>5}" \
            " | backfilling times: {:>5}".format(self.finish_time, cluster_utilization, average_wait_time,
                                                 self.job_wait_time_max, self.backfilling_time)
        print (s)

    def get_scheduling_decisions(self):
        return self.scheduling_decisions


if __name__ == "__main__":
    runCase = "ricc"
    print ("Loading the workloads and build the cluster...")
    target_workloads = Workloads("./data/RICC-2010-2.swf")
    target_cluster = Cluster(runCase, target_workloads.max_nodes, target_workloads.max_procs / target_workloads.max_nodes)
    print ("Finish loading the traces and building the cluster.")

    fcfs_test = False
    fafcfs_test = False
    slurm_test = False
    random_test = False
    sjf_test = True

    if fcfs_test:
        target_cluster.reset()
        print ("Start FCFS scheduling...")
        fcfs = Scheduler("fcfs")
        fcfs.schedule(target_cluster, target_workloads)
        print ("Finish FCFS scheduling and print results...")
        fcfs.print_scheduling_results(target_cluster)

        with open("data/fcfs_training_data.txt", "w") as f:
            for i in range(len(fcfs.scheduling_decisions)):
                _system_status = fcfs.scheduling_system_status[i]
                _job = fcfs.scheduling_decisions[i]
                _job_queue = _system_status["job_queue"]
                _free_nodes = _system_status["free_nodes"]
                _current_ts = _system_status["current_ts"]

                s = str(_free_nodes) + ","
                for [_, j] in _job_queue:
                    s += " {}, {}, {}, {}".format(j.job_id, (_current_ts - j.submit_time),
                                                  j.request_number_of_processors,
                                                  j.request_time)
                s += "\n"
                f.write(s)

                s = "{}, {}, {}, {}\n".format(_job.job_id, _job.submit_time,
                                              _job.request_number_of_processors, _job.request_time)
                f.write(s)

    if fafcfs_test:
        target_cluster.reset()
        print ("Start FirstAvailableFCFS scheduling...")
        fafcfs = Scheduler("small")
        fafcfs.schedule(target_cluster, target_workloads)
        print ("Finish FirstAvailableFCFS Scheduling and print results...")
        fafcfs.print_scheduling_results(target_cluster)

        with open("data/fafcfs_training_data.txt", "w") as f:
            for i in range(len(fafcfs.scheduling_decisions)):
                _system_status = fafcfs.scheduling_system_status[i]
                _job = fafcfs.scheduling_decisions[i]
                _job_queue = _system_status["job_queue"]
                _free_nodes = _system_status["free_nodes"]
                _current_ts = _system_status["current_ts"]

                s = str(_free_nodes) + ","
                for [_, j] in _job_queue:
                    s += " {}, {}, {}, {}".format(j.job_id, (_current_ts - j.submit_time),
                                                  j.request_number_of_processors,
                                                  j.request_time)
                s += "\n"
                f.write(s)

                s = "{}, {}, {}, {}\n".format(_job.job_id, _job.submit_time,
                                              _job.request_number_of_processors, _job.request_time)
                f.write(s)

    if slurm_test:
        target_cluster.reset()
        print ("Start Slurm scheduling...")
        slurm = Scheduler("slurm")
        slurm.configure_slurm(1000, 0, 1000, 0, 0, 0, 60 * 60 * 72, True)
        slurm.schedule(target_cluster, target_workloads)
        print ("Finish Slurm scheduling and print results...")
        slurm.print_scheduling_results(target_cluster)

        with open("data/slurm_training_data.txt", "w") as f:
            for i in range(len(slurm.scheduling_decisions)):
                _system_status = slurm.scheduling_system_status[i]
                _job = slurm.scheduling_decisions[i]
                _job_queue = _system_status["job_queue"]
                _free_nodes = _system_status["free_nodes"]
                _current_ts = _system_status["current_ts"]

                s = str(_free_nodes) + ","
                for [_, j] in _job_queue:
                    s += " {}, {}, {}, {}".format(j.job_id, (_current_ts - j.submit_time),
                                                  j.request_number_of_processors,
                                                  j.request_time)
                s += "\n"
                f.write(s)

                s = "{}, {}, {}, {}\n".format(_job.job_id, _job.submit_time,
                                              _job.request_number_of_processors, _job.request_time)
                f.write(s)

    if random_test:
        target_cluster.reset()
        print ("Start RandomSelect scheduling...")
        ran = Scheduler("random")
        ran.schedule(target_cluster, target_workloads)
        print ("Finish RandomSelect Scheduling and print results...")
        ran.print_scheduling_results(target_cluster)

        with open("data/random_training_data.txt", "w") as f:
            for i in range(len(ran.scheduling_decisions)):
                _system_status = ran.scheduling_system_status[i]
                _job = ran.scheduling_decisions[i]
                _job_queue = _system_status["job_queue"]
                _free_nodes = _system_status["free_nodes"]
                _current_ts = _system_status["current_ts"]

                s = str(_free_nodes) + ","
                for [_, j] in _job_queue:
                    s += " {}, {}, {}, {}".format(j.job_id, (_current_ts - j.submit_time),
                                                  j.request_number_of_processors,
                                                  j.request_time)
                s += "\n"
                f.write(s)

                s = "{}, {}, {}, {}\n".format(_job.job_id, _job.submit_time,
                                              _job.request_number_of_processors, _job.request_time)
                f.write(s)

    if sjf_test:
        target_cluster.reset()
        print ("Start Shortest Job First scheduling...")
        sjf = Scheduler("sjf")
        sjf.schedule(target_cluster, target_workloads)
        print ("Finish Shortest Job First Scheduling and print results...")
        sjf.print_scheduling_results(target_cluster)

        with open("data/sjf_training_data.txt", "w") as f:
            for i in range(len(sjf.scheduling_decisions)):
                _system_status = sjf.scheduling_system_status[i]
                _job = sjf.scheduling_decisions[i]
                _job_queue = _system_status["job_queue"]
                _free_nodes = _system_status["free_nodes"]
                _current_ts = _system_status["current_ts"]

                s = str(_free_nodes) + ","
                for [_, j] in _job_queue:
                    s += " {}, {}, {}, {}".format(j.job_id, (_current_ts - j.submit_time),
                                                  j.request_number_of_processors,
                                                  j.request_time)
                s += "\n"
                f.write(s)

                s = "{}, {}, {}, {}\n".format(_job.job_id, _job.submit_time,
                                              _job.request_number_of_processors, _job.request_time)
                f.write(s)