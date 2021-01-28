import re
import sys
import math


class Job:
    """
    1. Job Number -- a counter field, starting from 1.
    2. Submit Time -- in seconds. The earliest time the log refers to is zero, and is usually the submittal time of the first job. The lines in the log are sorted by ascending submittal times. It makes sense for jobs to also be numbered in this order.
    3. Wait Time -- in seconds. The difference between the job's submit time and the time at which it actually began to run. Naturally, this is only relevant to real logs, not to models.
    4. Run Time -- in seconds. The wall clock time the job was running (end time minus start time).
    We decided to use ``wait time'' and ``run time'' instead of the equivalent ``start time'' and ``end time'' because they are directly attributable to the Scheduler and application, and are more suitable for models where only the run time is relevant.
    Note that when values are rounded to an integral number of seconds (as often happens in logs) a run time of 0 is possible and means the job ran for less than 0.5 seconds. On the other hand it is permissable to use floating point values for time fields.
    5. Number of Allocated Processors -- an integer. In most cases this is also the number of processors the job uses; if the job does not use all of them, we typically don't know about it.
    6. Average CPU Time Used -- both user and system, in seconds. This is the average over all processors of the CPU time used, and may therefore be smaller than the wall clock runtime. If a log contains the total CPU time used by all the processors, it is divided by the number of allocated processors to derive the average.
    7. Used Memory -- in kilobytes. This is again the average per processor.
    8. Requested Number of Processors.
    9. Requested Time. This can be either runtime (measured in wallclock seconds), or average CPU time per processor (also in seconds) -- the exact meaning is determined by a header comment. In many logs this field is used for the user runtime estimate (or upper bound) used in backfilling. If a log contains a request for total CPU time, it is divided by the number of requested processors.
    10. Requested Memory (again kilobytes per processor).
    11. Status 1 if the job was completed, 0 if it failed, and 5 if cancelled. If information about chekcpointing or swapping is included, other values are also possible. See usage note below. This field is meaningless for models, so would be -1.
    12. User ID -- a natural number, between one and the number of different users.
    13. Group ID -- a natural number, between one and the number of different groups. Some systems control resource usage by groups rather than by individual users.
    14. Executable (Application) Number -- a natural number, between one and the number of different applications appearing in the workload. in some logs, this might represent a script file used to run jobs rather than the executable directly; this should be noted in a header comment.
    15. Queue Number -- a natural number, between one and the number of different queues in the system. The nature of the system's queues should be explained in a header comment. This field is where batch and interactive jobs should be differentiated: we suggest the convention of denoting interactive jobs by 0.
    16. Partition Number -- a natural number, between one and the number of different partitions in the systems. The nature of the system's partitions should be explained in a header comment. For example, it is possible to use partition numbers to identify which machine in a cluster was used.
    17. Preceding Job Number -- this is the number of a previous job in the workload, such that the current job can only start after the termination of this preceding job. Together with the next field, this allows the workload to include feedback as described below.
    18. Think Time from Preceding Job -- this is the number of seconds that should elapse between the termination of the preceding job and the submittal of this one.
    """
    def __init__(self, line = "0        0      0    0   0     0    0   0  0 0  0   0   0  0  0 0 0 0"):
        line = line.strip()
        s_array = re.split("\\s+", line)
        self.job_id = int(s_array[0])
        self.submit_time = int(s_array[1])
        self.wait_time = int(s_array[2])
        self.run_time = int(s_array[3])
        self.number_of_allocated_processors = int(s_array[4])
        self.average_cpu_time_used = float(s_array[5])
        self.used_memory = int(s_array[6])

        # "requested number of processors" and "number of allocated processors" are typically mixed.
        # I do not know their difference clearly. But it seems to me using a larger one will be sufficient.
        self.request_number_of_processors = int(s_array[7])
        self.number_of_allocated_processors = max(self.number_of_allocated_processors, self.request_number_of_processors)
        self.request_number_of_processors = self.number_of_allocated_processors
        
        self.request_number_of_nodes = -1
        
        # if we use the job's request time field
        # for model, request_time might be empty. In this case, we set request_time to the run_time
        self.request_time = int(s_array[8])
        if self.request_time == -1:
            self.request_time = self.run_time

        # if we use the run time as the most accurate request time
        # self.request_time = self.run_time + 60
        # if we gradually increase the accuracy of job's request time
        # with a percentage wrong estimation and round to a fixed time: 1,2,3,... hours.
        # this.requestTime = (int) (this.runTime + this.runTime * 0.4);
        # int roundsTo = 60 * 60; //round up to hours
        # this.requestTime = (this.requestTime / roundsTo + 1) * roundsTo;

        self.request_memory = int(s_array[9])
        self.status = int(s_array[10])
        self.user_id = int(s_array[11])
        self.group_id = int(s_array[12])
        self.executable_number = int(s_array[13])
        self.queue_number = int(s_array[14])

        try:
            self.partition_number = int(s_array[15])
        except ValueError:
            self.partition_number = 0

        self.proceeding_job_number = int(s_array[16])
        self.think_time_from_proceeding_job = int(s_array[17])

        self.random_id = self.submit_time

        self.scheduled_time = -1

        self.allocated_machines = None

        self.slurm_in_queue_time = 0
        self.slurm_age = 0
        self.slurm_job_size = 0.0
        self.slurm_fair = 0.0
        self.slurm_partition = 0
        self.slurm_qos = 0
        self.slurm_tres_cpu = 0.0
        
    def __eq__(self, other):
        return self.job_id == other.job_id

    def __lt__(self, other):
        return self.job_id < other.job_id

    def __hash__(self):
        return hash(self.job_id)

    def __str__(self):
        return "J["+str(self.job_id)+"]-["+str(self.request_number_of_processors)+"]-["+str(self.submit_time)+"]-["+str(self.request_time)+"]"
    def __feature__(self):
        return [self.submit_time, self.request_number_of_processors, self.request_time,
                self.user_id, self.group_id, self.executable_number, self.queue_number]


class Workloads:

    def __init__(self, path):
        self.all_jobs = []
        self.max = 0
        self.max_exec_time = 0
        self.min_exec_time = sys.maxsize
        self.max_job_id = 0

        self.max_requested_memory = 0
        self.max_user_id = 0
        self.max_group_id = 0
        self.max_executable_number = 0
        self.max_job_id = 0
        self.max_nodes = 0
        self.max_procs = 0

        with open(path) as fp:
            for line in fp:
                if line.startswith(";"):
                    if line.startswith("; MaxNodes:"):
                        self.max_nodes = int(line.split(":")[1].strip())
                    if line.startswith("; MaxProcs:"):
                        self.max_procs = int(line.split(":")[1].strip())
                    continue

                j = Job(line)
                if j.run_time > self.max_exec_time:
                    self.max_exec_time = j.run_time
                if j.run_time < self.min_exec_time:
                    self.min_exec_time = j.run_time
                if j.request_memory > self.max_requested_memory:
                    self.max_requested_memory = j.request_memory
                if j.user_id > self.max_user_id:
                    self.max_user_id = j.user_id
                if j.group_id > self.max_group_id:
                    self.max_group_id = j.group_id
                if j.executable_number > self.max_executable_number:
                    self.max_executable_number = j.executable_number

                # filter those illegal data whose runtime < 0
                if j.run_time < 0:
                    j.run_time = 10
                if j.run_time > 0:
                    self.all_jobs.append(j)
                
                    if j.request_number_of_processors > self.max:
                        self.max = j.request_number_of_processors

        # if max_procs = 0, it means node/proc are the same.
        if self.max_procs == 0:
            self.max_procs = self.max_nodes

        print ("Max Allocated Processors:", str(self.max), ";max node:", self.max_nodes,
               ";max procs:", self.max_procs,
               ";max execution time:", self.max_exec_time)

        self.all_jobs.sort(key=lambda job: job.job_id)

    def size(self):
        return len(self.all_jobs)

    def reset(self):
        for job in self.all_jobs:
            job.scheduled_time = -1

    def __getitem__(self, item):
        return self.all_jobs[item]


if __name__ == "__main__":
    print ("Loading the workloads...")
    load = Workloads("../../../data/lublin_256.swf")
    print ("Finish loading the workloads...", type(load[0]))
    print (load.max_nodes, load.max_procs)
    print (load[0].__feature__())
    print (load[1].__feature__())
    
    # empty_job_str = "0        0      0    0   0     0    0   0  0 0  0   0   0  0  0 0 0 0"
    # empty_job = Job(empty_job_str)
    # print (empty_job.job_id, empty_job.feature())