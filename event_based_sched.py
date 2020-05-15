import heapq
from collections import namedtuple
from job import Workloads
from cluster import Cluster
from gym.utils import seeding
from matplotlib import pyplot as plt
import time
from collections import deque
from utils import *


Event = namedtuple("Event", ["timestamp", "name", "payload"])
algos = [fcfs_score, wfp_score, uni_score, sjf_score, f1_score]

class EventManager:
    def __init__(self, workload_file="data/lublin_256.swf", start=0, sequence_length=256, algos=algos):
        self.workload = Workloads(workload_file)
        self.cluster = Cluster("Cluster", self.workload.max_nodes, self.workload.max_procs/self.workload.max_nodes)
        self.start = start
        self.sequence_length = sequence_length
        self.current_timestamp = 0
        self.done = 0
        self.remain = self.sequence_length
        self.deciding = 0
        self.algos = algos
        self.algo = fcfs_score
        self.reward = []
        self.reset(start)

    def reset(self, start=0, algo=fcfs_score):
        self.workload.reset()
        self.start = start
        self.running_jobs = []
        self.job_queue = deque()
        self.finished = []
        self.__event_queue = []
        self.reward = []
        self.current_timestamp = 0
        self.done = 0
        self.remain = self.sequence_length
        self.deciding = 0
        self.algo = algo
        for offset in range(self.sequence_length):
            job = self.workload[self.start + offset]
            event = Event(job.submit_time, "SubmitJob", job)
            self.register(event)

    def performance_of_algos(self):
        np_random, seed = seeding.np_random(0)
        all_data = [[] for _ in range(len(self.algos))]
        for j in range(10):
            s = np_random.randint(0, (self.workload.size() - self.sequence_length - 1))
            for i, algo in enumerate(self.algos):
                self.reset(s, algo)
                while self.done != 1:
                    self.dispatch()
                all_data[i].append(sum(self.reward)/self.sequence_length)
        xticks = [y + 1 for y in range(len(all_data))]
        for y in range(6):
            plt.plot(xticks[y:y+1], all_data[y:y+1], 'o', color='darkorange')

        plt.boxplot(all_data, showfliers=False, meanline=True, showmeans=True, medianprops={"linewidth": 0},
                    meanprops={"color": "darkorange", "linewidth": 4, "linestyle": "solid"})

        plt.show()

    def run_one_trajectory(self):
        self.reset(0, fcfs_score)
        while self.done != 1:
            self.dispatch()

    def dispatch(self):
        event = heapq.heappop(self.__event_queue)
        timestamp, name, payload = event
        # print(timestamp, name, payload, len(self.job_queue), len(self.running_jobs), len(self.finished))
        assert timestamp >= self.current_timestamp
        self.current_timestamp = timestamp
        if name == "SubmitJob":
            self.handleSubmitJob(payload)
        elif name == "ScheduleJob":
            self.handleScheduleJob(payload)
        elif name == "SfinishJob":
            self.handleFinishJob(payload)
        elif name == "Done":
            self.handleDone(payload)
        else:
            raise NotImplementedError

    def register(self, event):
        try:
            heapq.heappush(self.__event_queue, event)
        except:
            print(event)
            raise Exception

    def job_score(self, job_for_scheduling):
        assert job_for_scheduling.run_time > 0 or print(job_for_scheduling)
        _tmp = max(1.0, (float(
            job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)
                         /
                         max(job_for_scheduling.run_time, 10)))
        return _tmp

    def canSchedule(self):
        return self.job_queue

    def isDone(self):
        return self.remain == 0 and not self.job_queue and not self.running_jobs and not self.deciding

    def handleDone(self, payload):
        assert len(self.__event_queue)==0
        self.done = 1

    def handleSubmitJob(self, payload):
        job = payload
        insort(self.job_queue, job, key=self.algo)
        self.remain -= 1
        if not self.deciding and self.canSchedule():
            next_event = Event(job.submit_time, "ScheduleJob", self.job_queue[0])
            self.register(next_event)

    def handleScheduleJob(self, payload):
        self.deciding = 1
        assert self.job_queue
        job_scheduling = self.job_queue.popleft()
        while not self.cluster.can_allocated(job_scheduling):
            self.dispatch()
        job_scheduling.allocated_machines =  self.cluster.allocate(None, job_scheduling.request_number_of_processors)
        self.running_jobs.append(job_scheduling)
        assert job_scheduling.scheduled_time == -1
        job_scheduling.scheduled_time = self.current_timestamp
        self.reward.append(self.job_score(job_scheduling))
        next_event = Event(self.current_timestamp+job_scheduling.run_time, "SfinishJob", job_scheduling)
        self.register(next_event)
        if self.canSchedule():
            next_event = Event(self.current_timestamp, "ScheduleJob", self.job_queue[0])
            self.register(next_event)
        self.deciding = 0


    def handleFinishJob(self, payload):
        job_finishing = payload
        self.running_jobs.remove(job_finishing)
        self.finished.append(job_finishing)
        self.cluster.release(job_finishing.allocated_machines)
        assert job_finishing.scheduled_time != -1
        if self.isDone():
            next_event = Event(self.current_timestamp, "Done", None)
            self.register(next_event)
        else:
            if not self.deciding and self.canSchedule():
                next_event = Event(self.current_timestamp, "ScheduleJob", self.job_queue[0])
                self.register(next_event)

if __name__ == '__main__':
    em = EventManager(workload_file='data/HPC2N-2002-2.2-cln.swf', sequence_length=1024)
    start = time.time()
    em.performance_of_algos()
    # em.run_one_trajectory()
    print("Using: ", time.time()-start)



