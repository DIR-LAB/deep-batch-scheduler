import numpy as np
import math
import sys
import os
from six import StringIO, b

import gym
from gym.utils import seeding
from gym.envs.scheduler.job import Job
from gym.envs.scheduler.job import Workloads
from gym.envs.scheduler.cluster import Cluster
from gym.envs.scheduler.cluster import Machine

# HPC Batch Scheduler Simulation.
# 
# Created by Dong Dai. Licensed on the same terms as the rest of OpenAI Gym.

MAX_QUEUE_SIZE = 50
MAX_WAIT_TIME = 24 * 60 * 60 # assume maximal wait time is 24 hours.
MAX_RUN_TIME = 12 * 60 * 60 # assume maximal runtime is 12 hours
NUM_JOB_SCHEDULERS = 4

# each job has features
# submit_time, request_number_of_processors, request_time,
# user_id, group_id, executable_number, queue_number
JOB_FEATURES = 7

MAX_JOBS_EACH_BATCH = 300

class HpcEnv(gym.Env):
    """The main HPC Environment class. 
    The main API methods that users of this class need to know are:

        step
        reset
        render
        close
        seed

    And set the following attributes:

        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards

    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.

    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality over time.
    """

    metadata = {
        'render.modes': ['human','ansi']
    }

    def __init__(self, workload_file = ''):
        super(HpcEnv, self).__init__()

        print ("loading workloads from dataset:", workload_file)
        self.loads = Workloads(workload_file)
        self.cluster = Cluster("Ricc", self.loads.max_nodes, self.loads.max_procs/self.loads.max_nodes)

        self.seed(1)

        self.action_space = NUM_JOB_SCHEDULERS
        self.observation_space = JOB_FEATURES * MAX_QUEUE_SIZE + 1

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

        
        self.n_actions = MAX_QUEUE_SIZE
        self.n_features = MAX_QUEUE_SIZE * JOB_FEATURES
    
    def render(self, mode='human'):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        ''' 
        # This code snippet is from frozonLake-v0. 
        # We need to use the similar tech to viz our environment
        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")
        '''
        if mode != 'human':
            return outfile


    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
        return 0

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.

        This is the place we formulate a random starting situation. 
        There are certain rules to create it: 
        1) randomly start from a job in the workload
        2) randomly create running jobs to fill the queue
        """
        for i in range(MAX_QUEUE_SIZE):
            self.job_queue[i] = Job()

        self.start = self.np_random.randint(self.loads.size())
        self.current_point = self.start

        # the size of jobs scheduled in this batch
        job_remainder = self.loads.size() - self.start # how many jobs are remainded in the workload
        self.size = self.np_random.randint(min(job_remainder, MAX_JOBS_EACH_BATCH))

        self.current_timestamp = self.loads[self.start].submit_time
        self.start_timestamp = self.current_timestamp

        # Generate random running jobs to fill the cluster.
        # From a random starter, the running jobs and their resources utilization can be 
        # anything. We need to fully generate them to train the agent.
        # Strategy 1: random number of jobs + random request_number_of_processors + random run_time,
        running_job_size = self.np_random.randint(MAX_JOBS_EACH_BATCH)
        for i in range(running_job_size):
            req_num_of_processors = self.np_random.randint(self.loads.max_procs)
            runtime_of_job = self.np_random.randint(self.loads.max_exec_time)
            job_tmp = Job()
            job_tmp.job_id = (0 - i) # to be different from the normal jobs
            job_tmp.request_number_of_processors = req_num_of_processors
            job_tmp.run_time = runtime_of_job
            if self.cluster.can_allocated(job_tmp):
                self.running_jobs.append(job_tmp)
                # assume job was randomly generated
                job_tmp.scheduled_time = (self.current_timestamp - self.np_random.randint(runtime_of_job))
                job_tmp.allocated_machines = self.cluster.allocate(job_tmp.job_id, job_tmp.request_number_of_processors)
            else:
                break

        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))

        #self.cluster.free_node
        job_queue_vec = np.zeros(JOB_FEATURES * MAX_QUEUE_SIZE)
        job_queue_vec[0:JOB_FEATURES+1] = self.loads[self.start].feature()
        return np.append(job_queue_vec, self.cluster.free_node)


    def build_queue_vector(self):
        vector = np.array([])

        for job in self.job_queue:

            if job.job_id == 0: # empty job
                for i in range(JOB_FEATURES):
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
            normalized_wait_time = min(1, float(wait_time) / float(MAX_WAIT_TIME))
            normalized_request_time = min(1, float(request_time) / float(MAX_RUN_TIME))

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
            if self.loads[i].scheduled_time == 0:
                return False
        return True

    def is_job_queue_empty(self):
        return all(v.job_id == 0 for v in self.job_queue)


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
            first_to_submit = self.loads[self.current_point].submit_time

        if more_jobs_to_schedule and more_jobs_running:
            if first_to_submit <= first_to_release_time:
                for i in range(0, len(self.job_queue)):
                    if self.job_queue[i].job_id == 0:
                        self.job_queue[i] = self.loads[self.current_point]
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
                    self.job_queue[i] = self.loads[self.current_point]
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

        self.job_queue[action] = Job()  # change it to an empty job
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
                delay = self.loads[i].scheduled_time - self.loads[i].submit_time
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
                        float(self.loads[i].request_number_of_processors) / float(self.cluster.num_procs_per_node)))
                consumed_node_hours += self.loads[i].run_time * request_nodes

            for i in range(index, index + num):
                finish = self.loads[i].scheduled_time + self.loads[i].run_time

                request_nodes = int(
                    math.ceil(float(self.loads[i].request_number_of_processors) / float(self.cluster.num_procs_per_node)))

                consumed_node_hours += self.loads[i].run_time * request_nodes

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
    print(o.seed(),o.np_random.randint(1, o.loads.size()))
