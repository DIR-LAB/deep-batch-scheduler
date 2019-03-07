import tensorflow as tf

import sys
import os
import numpy as np
import math
import json
import random
import time
import scipy

from job import Job
from job import Workloads
from cluster import Cluster

MAX_QUEUE_SIZE = 64
MAX_JOBS_EACH_BATCH = 64
MIN_JOBS_EACH_BATCH = 64
MAX_MACHINE_SIZE = 1024
MAX_WAIT_TIME = 12 * 60 * 60 # assume maximal wait time is 12 hours.
MAX_RUN_TIME = 12 * 60 * 60 # assume maximal runtime is 12 hours

SEED = 42

# each job has three features: submit_time, request_number_of_processors, request_time/run_time,
JOB_FEATURES = 3
DEBUG = False

BN_EPSILON = 0.001
weight_decay = 0.0002  # scale for l2 regularization
num_residual_blocks = 5 # How many residual blocks do you want


def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''

    ## TODO: to allow different weight decay to fully connected layer and conv layer
    regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables

def output_layer(input_layer, num_labels):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h

def batch_normalization_layer(input_layer, dimension):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                           initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                            initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer

def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output

def bn_relu_conv_layer(input_layer, filter_shape, stride):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer

def residual_block(input_layer, output_channel, first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                      input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output

def inference(input_tensor_batch, act_dim, n, reuse):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''

    layers = []
    with tf.variable_scope('conv0', reuse=reuse):
        conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1)
        activation_summary(conv0)
        layers.append(conv0)

    for i in range(n):
        with tf.variable_scope('conv1_%d' % i, reuse=reuse):
            if i == 0:
                conv1 = residual_block(layers[-1], 16, first_block=True)
            else:
                conv1 = residual_block(layers[-1], 16)
            activation_summary(conv1)
            layers.append(conv1)

    for i in range(n):
        with tf.variable_scope('conv2_%d' % i, reuse=reuse):
            conv2 = residual_block(layers[-1], 32)
            activation_summary(conv2)
            layers.append(conv2)

    for i in range(n):
        with tf.variable_scope('conv3_%d' % i, reuse=reuse):
            conv3 = residual_block(layers[-1], 64)
            layers.append(conv3)
        assert conv3.get_shape().as_list()[1:] == [34, 2, 64]

    with tf.variable_scope('fc', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layers[-1], in_channel)
        relu_layer = tf.nn.relu(bn_layer)
        global_pool = tf.reduce_mean(relu_layer, [1, 2])

        print(global_pool.get_shape().as_list()[-1:])
        assert global_pool.get_shape().as_list()[-1:] == [64]
        output = output_layer(global_pool, act_dim)
        layers.append(output)

    return layers[-1]


class HPCEvn():
    def __init__(self):
        self.act_dim = MAX_QUEUE_SIZE
        self.obs_dim = MAX_QUEUE_SIZE + MAX_MACHINE_SIZE

        # initialize job queue
        self.job_queue = []
        for i in range(0, MAX_QUEUE_SIZE):
            self.job_queue.append(Job())

        self.running_jobs = []
        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0

        # initialize random state used by the whole system.
        random.seed(SEED)

        self.loads = None
        self.cluster = None
        self.bsld_fcfs_dict = {}

    def actual_init(self, workload_file = ''):
        print("loading workloads from:", workload_file)
        self.loads = Workloads(workload_file)
        self.cluster = Cluster("cluster", self.loads.max_nodes, self.loads.max_procs / self.loads.max_nodes)

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
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0

        # randomly sample a sequence of jobs from workload
        self.start = random.randint(0, (self.loads.size() - MAX_JOBS_EACH_BATCH))
        self.num_job_in_batch = random.randint(MIN_JOBS_EACH_BATCH, MAX_JOBS_EACH_BATCH)
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue[0] = self.loads[self.start]
        self.next_arriving_job_idx = self.start + 1

        # schedule the sequence of jobs using FCFS. This would be the standard references for this sequence.
        self.bsld_fcfs_dict = {}

        while True:
            self.job_queue.sort(key=lambda j:(j.submit_time))
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
                    _tmp = max(1.0, (float(job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)
                                     /
                                     max(job_for_scheduling.run_time, 10)))
                    self.bsld_fcfs_dict[job_for_scheduling_index] = (_tmp / self.num_job_in_batch)
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

    def step(self, action):
        # action is a legal job ready for scheduling.
        assert self.job_queue[action].job_id != 0 # this job should be legal.
        job_for_scheduling = self.job_queue[action]
        job_for_scheduling_index = action

        get_this_job_scheduled = False
        bsld = 0.0
        scheduled_logs = []

        if self.cluster.can_allocated(job_for_scheduling):
            assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
            job_for_scheduling.scheduled_time = self.current_timestamp
            if DEBUG:
                print("In step, schedule a job, ", job_for_scheduling, " with free nodes: ", self.cluster.free_node)
            job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                          job_for_scheduling.request_number_of_processors)
            self.running_jobs.append(job_for_scheduling)
            scheduled_logs.append(job_for_scheduling)
            get_this_job_scheduled = True
            _tmp = max(1.0, (float(job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)
                              /
                              max(job_for_scheduling.run_time, 10)))
            bsld += (_tmp / self.num_job_in_batch)
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
                    scheduled_logs.append(_job)
                    _tmp = max(1.0, float(_job.scheduled_time - _job.submit_time + _job.run_time)
                                /
                                max(_job.run_time, 10))
                    bsld += (_tmp / self.num_job_in_batch)
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

        # we want to minimize bsld. Instead of using Mao's method, we first tried our own design.
        # reward = AverageBSLD_Scheduled(FCFS) - AverageBSLD_Scheduled(Our).
        fcfs = 0.0
        for _job in scheduled_logs:
            fcfs += (self.bsld_fcfs_dict[_job._job.job_id])
        reward = fcfs - bsld

        done = True
        for i in range(self.start, self.last_job_in_batch):
            if self.loads[i].scheduled_time == -1:  # have at least one job in the batch who has not been scheduled
                done = False
                break

        return [obs, reward, done, None]


def resnet(x_ph, act_dim):
    '''
    https://github.com/wenxinxu/resnet-in-tensorflow
    '''
    x = tf.reshape(x_ph, shape=[-1, 136, 8, 3])
    return inference(x, act_dim, num_residual_blocks, reuse=False)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def statistics_scalar(x):
    x = np.array(x, dtype=np.float32)
    sum, n = [np.sum(x), len(x)]
    mean = sum / n
    sq = (np.sum((x - mean) ** 2))
    std = np.sqrt(sq / n)
    return mean, std

class Buffer:
    """
    A buffer for storing trajectories experienced by a HPC agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf]


"""
REINFORCE RL Algorithm
"""
def reinforce(workload_file, output_dir, steps_per_epoch=4000, epochs=50,
              gamma=1, pi_lr=3e-4, vf_lr=1e-5, train_v_iters=80,
              lam=0.97, max_ep_len=10000, save_freq=10):
    hpc = HPCEvn()
    hpc.actual_init(workload_file)
    model_file = os.path.join(output_dir, "nn")

    obs_dim = hpc.obs_dim
    act_dim = hpc.act_dim

    # Inputs to compute graphs
    x_ph = tf.placeholder(dtype=tf.float32, shape=(None, obs_dim))
    a_ph = tf.placeholder(dtype=tf.int32, shape=(None, ))
    adv_ph = tf.placeholder(dtype=tf.float32, shape=(None, ))
    ret_ph = tf.placeholder(dtype=tf.float32, shape=(None, ))
    logp_old_ph = tf.placeholder(dtype=tf.float32, shape=(None, ))

    # Main outputs from the computation graph
    logits = resnet(x_ph, act_dim)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits, 1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a_ph, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)

    # value network
    v = tf.squeeze(resnet(x_ph, 1), axis=1)

    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    # every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]

    # TODO: VPGBuffer
    buf = Buffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    # Objectives
    pi_loss = -tf.reduce_mean(logp * adv_ph)
    v_loss = tf.reduce_mean((ret_ph - v) ** 2)

    # Info (useful during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp)
    approx_ent = tf.reduce_mean(-logp)

    # Optimizer
    train_pi = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    train_v = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    def update():
        inputs = {k: v for k, v in zip(all_phs, buf.get())}
        pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)

        sess.run(train_pi, feed_dict=inputs)

        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)

        pi_l_new, v_l_new, kl = sess.run([pi_loss, v_loss, approx_kl], feed_dict=inputs)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = hpc.reset(), 0, False, 0, 0

    # main loop
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            a, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1, -1)})
            buf.store(o, a, r, v_t, logp_t)

            o, r, d, _ = hpc.step(a[0])
            ep_ret += r
            ep_len += 1

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t == steps_per_epoch - 1):
                if not (terminal):
                    print("Warning: trajectory cut off by epoch at %d steps." % ep_len)
                last_val = r if d else sess.run(v, feed_dict={x_ph: o.reshape(1, -1)})
                buf.finish_path(last_val)
                o, r, d, ep_ret, ep_len = hpc.reset(), 0, False, 0, 0

        # save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            save_path = saver.save(sess, model_file)

        update()


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: python reinforce.py workload_file output_model_dir")
        sys.exit()

    '''
    # model save/restore: https://blog.csdn.net/huachao1001/article/details/78501928
    # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    nn_model_dir = sys.argv[1]
    nn_model_file = os.path.join(nn_model_dir, "nn.meta")
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(nn_model_file)
        saver.restore(sess, tf.train.latest_checkpoint(nn_model_dir))
        graph = tf.get_default_graph()
        # w1 = graph.get_tensor_by_name("w1:0")
        # w2 = graph.get_tensor_by_name("w2:0")
        # feed_dict = {w1: 13.0, w2: 17.0}
        # op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
        # print(sess.run(op_to_restore, feed_dict))
    '''
    workload_file = sys.argv[1] # "../../data/RICC-2010-2.swf"
    output_model_dir = sys.argv[2] # "../../data/logs/hpc-reinforce"
    print("workload file:", workload_file, "output model:", output_model_dir)

    reinforce(workload_file, output_model_dir)


