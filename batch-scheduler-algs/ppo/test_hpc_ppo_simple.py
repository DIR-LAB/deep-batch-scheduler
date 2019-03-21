import time
import joblib
import os
import os.path as osp
import tensorflow as tf
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph

import gym
import hpc
import random
import math
import numpy as np
import sys

MAX_QUEUE_SIZE = 63
MAX_JOBS_EACH_BATCH = 63
JOB_FEATURES = 3


def load_policy(fpath, env_name, workload_file, itr='last'):
    # handle which epoch to load from
    if itr=='last':
        saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'+itr))

    # get the correct op for executing actions
    action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x.reshape(1, -1)}) # x[None,:]

    # initialize the environment from scratch
    env = gym.make(env_name)
    env.my_init(workload_file=workload_file)

    return env, get_action

def smalljf_get_action(obs):
    jobs = []
    for i in range(0, MAX_QUEUE_SIZE):
        normalized_request_nodes = obs[0][i * JOB_FEATURES + 2]
        if normalized_request_nodes == 0:
            jobs.append(-1)
        else:
            jobs.append(1 - normalized_request_nodes)  # normalized_run_time
    return [np.argmax(jobs)]

def sjf_get_action(obs):
    jobs = []
    for i in range(0, MAX_QUEUE_SIZE):
        run_time = obs[0][i * JOB_FEATURES + 1]
        if run_time == 0:
            jobs.append(-1)
        else:
            jobs.append(1 - run_time)  # normalized_run_time
    return [np.argmax(jobs)]

def fcfs_get_action(obs):
    jobs = []
    for i in range(0, MAX_QUEUE_SIZE):
        jobs.append(obs[0][i * JOB_FEATURES])  # normalized_wait_time
    return [np.argmax(jobs)]

def run_policy(env, get_action, max_ep_len=None, num_episodes=1, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset_for_test(), 0, False, 0, 0, 0
    while True:
        # a = get_action(o)
        a = sjf_get_action(o)
        # a = fcfs_get_action(o)
        # a = smalljf_get_action(o)
        o, r, d, scheduled = env.step_for_test(a)
        if scheduled:
            print(0 - r)
        if d:
            break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default='../../data/models/hpc-ppo-simple-direct-162k-Q35-empty-mpi/hpc-ppo-simple-direct-162k-Q35-empty-mpi_s1/')
    parser.add_argument('--env', type=str, default='Scheduler-v5')
    parser.add_argument('--workload', type=str, default='../../data/lublin_256.swf')
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    args = parser.parse_args()

    random.seed(1)

    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)

    env, get_action = load_policy(args.fpath, args.env, workload_file, args.itr if args.itr >=0 else 'last')
    run_policy(env, get_action, args.len, args.episodes, False)