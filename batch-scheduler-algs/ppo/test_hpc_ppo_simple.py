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

import matplotlib.pyplot as plt
plt.rcdefaults()


MAX_QUEUE_SIZE = 35
MAX_JOBS_EACH_BATCH = 10*32
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
    v_op = model['v']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x.reshape(1, -1)}) # x[None,:]
    get_value = lambda x : sess.run(v_op, feed_dict={model['x']: x.reshape(1, -1)})

    # initialize the environment from scratch
    env = gym.make(env_name)
    env.my_init(workload_file=workload_file)

    return env, get_action, get_value

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

def random_get_action(obs):
    return [random.randint(0, MAX_QUEUE_SIZE)]

def f1_get_action(orig_obs):
    jobs = []
    for i in range(0, MAX_QUEUE_SIZE):
        [submit_time, run_time, request_processors] = orig_obs[0][i * JOB_FEATURES : (i+1) * JOB_FEATURES]
        if submit_time == 0 and run_time == 0 and request_processors == 0:
            jobs.append(-1)
        else:
            # f1: log10(r)*n + 8.70 * 100 * log10(s)
            f1_score = sys.maxsize - (np.log10(request_processors) * run_time + 870 * np.log10(submit_time))
            jobs.append(f1_score)
    return [np.argmax(jobs)]

def f2_get_action(orig_obs):
    jobs = []
    for i in range(0, MAX_QUEUE_SIZE):
        [submit_time, run_time, request_processors] = orig_obs[0][i * JOB_FEATURES : (i+1) * JOB_FEATURES]
        if submit_time == 0 and run_time == 0 and request_processors == 0:
            jobs.append(-1)
        else:
            # f2: r^(1/2)*n + 25600 * log10(s)
            f2_score = sys.maxsize - (np.sqrt(run_time) * request_processors + 25600 * np.log10(submit_time))
            jobs.append(f2_score)
    return [np.argmax(jobs)]


def run_policy(env, get_action, get_value, max_ep_len=None, num_episodes=1, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    number_of_better = 0
    number_of_best = 0
    fcfs_r = []
    rl_r = []
    model_r = []
    sjf_r = []
    f1_r = []
    f2_r = []

    random.seed()
    for i in range(0, 10):
        # start = random.randint(MAX_JOBS_EACH_BATCH, (env.loads.size() - 2 * MAX_JOBS_EACH_BATCH)) # i + MAX_JOBS_EACH_BATCH
        # nums = MAX_JOBS_EACH_BATCH # random.randint(MAX_JOBS_EACH_BATCH, MAX_JOBS_EACH_BATCH)
        start = random.randint(300, 8000)
        nums = 1500 #env.loads.size() - 2 * MAX_JOBS_EACH_BATCH

        model = 0
        rl = 0
        fcfs = 0
        sjf = 0
        f1 = 0
        f2 = 0
        m = 0
        s = 0

        o, r, d, ep_ret, ep_len, n = env.reset_for_test(start, nums), 0, False, 0, 0, 0
        while True:
            a = get_action(o)
            o, r, d, scheduled = env.step_for_test(a)
            if d:
                rl = 0 - r
                break

        same = 0
        total = 0
        o, r, d, ep_ret, ep_len, n = env.reset_for_test(start, nums), 0, False, 0, 0, 0
        while True:
            v = get_value(o)
            a_m = get_action(o)
            a_s = sjf_get_action(o)
            if a_m == a_s:
                same += 1
            total += 1
            # a = random_get_action(o)
            # a = fcfs_get_action(o)
            # a = smalljf_get_action(o)

            if v < -0.5:
                a = a_s
                s += 1
            else:
                a = a_m
                m += 1
            o, r, d, scheduled = env.step_for_test(a)
            #if scheduled:
            #    print(0 - r, v)
            if d:
                # print (0 -r, end=" ")
                model = 0 - r
                break
        # print ("Ratio: ", same / total)

        o, r, d, ep_ret, ep_len, n = env.reset_for_test(start, nums), 0, False, 0, 0, 0
        while True:
            a = fcfs_get_action(o)
            o, r, d, scheduled = env.step_for_test(a)
            if d:
                fcfs = 0 - r
                break

        o, r, d, ep_ret, ep_len, n = env.reset_for_test(start, nums), 0, False, 0, 0, 0
        while True:
            a = sjf_get_action(o)
            o, r, d, scheduled = env.step_for_test(a)
            if d:
                sjf = 0 - r
                break

        # F1 obs should be the origin observation
        o, r, d, ep_ret, ep_len, n = env.reset_for_test(start, nums, True), 0, False, 0, 0, 0
        while True:
            a = f1_get_action(o)
            o, r, d, scheduled = env.step_for_test(a, True)
            if d:
                f1 = 0 - r
                break

        # F2 obs should be the origin observation
        o, r, d, ep_ret, ep_len, n = env.reset_for_test(start, nums, True), 0, False, 0, 0, 0
        while True:
            a = f2_get_action(o)
            o, r, d, scheduled = env.step_for_test(a, True)
            if d:
                f2 = 0 - r
                break

        print("iteration:%4d start:%4d nums:%4d \t %4.5f \t %4.5f \t %4.5f \t %4.5f \t %4.5f \t %4.5f \t %4.5f"% (i, start, nums, fcfs, rl, model, sjf, f2, f1, (m / (m + s))))

        model_r.append(model)
        sjf_r.append(sjf)
        fcfs_r.append(fcfs)
        f1_r.append(f1)
        f2_r.append(f2)
        rl_r.append(rl)

        if model <= 1.1 * sjf:
            number_of_better += 1

        if model <= 1.1 * f1:
            number_of_best += 1
    print("better number:", number_of_better, "best number:", number_of_best)

    # plot
    all_data = []
    all_data.append(fcfs_r)
    all_data.append(rl_r)
    all_data.append(model_r)
    all_data.append(sjf_r)
    all_data.append(f2_r)
    all_data.append(f1_r)

    all_medians = []
    for p in all_data:
        all_medians.append(np.median(p))

    plt.rc("font", size=35)
    plt.figure(figsize=(16, 14))
    axes = plt.axes()

    xticks = [y + 1 for y in range(len(all_data))]
    plt.plot(xticks[0:1], all_data[0:1], 'o', color='darkorange')
    plt.plot(xticks[1:2], all_data[1:2], 'o', color='darkorange')
    plt.plot(xticks[2:3], all_data[2:3], 'o', color='darkorange')
    plt.plot(xticks[3:4], all_data[3:4], 'o', color='darkorange')
    plt.plot(xticks[4:5], all_data[4:5], 'o', color='darkorange')
    plt.plot(xticks[5:6], all_data[5:6], 'o', color='darkorange')

    plt.boxplot(all_data, showfliers=False)

    axes.yaxis.grid(True)
    axes.set_xticks([y + 1 for y in range(len(all_data))])
    xticklabels = ['FCFS', 'RL', 'Model', 'SJF', 'F2', 'F1']
    plt.setp(axes, xticks=[y + 1 for y in range(len(all_data))],
             xticklabels=['FCFS', 'RL', 'Model', 'SJF', 'F2', 'F1'])

    plt.tick_params(axis='both', which='major', labelsize=45)
    plt.tick_params(axis='both', which='minor', labelsize=45)

    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default='../../data/models/hpc-ppo-simple-162k-q35-empty-mpi-v2/hpc-ppo-simple-162k-q35-empty-mpi-v2_s1/')
    # parser.add_argument('--fpath', type=str, default='../../data/models/hpc-ppo-simple-direct-162k-Q35-empty-mpi/hpc-ppo-simple-direct-162k-Q35-empty-mpi_s1/')
    parser.add_argument('--env', type=str, default='Scheduler-v5')
    parser.add_argument('--workload', type=str, default='../../data/lublin_256.swf')
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    args = parser.parse_args()

    random.seed(1)

    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)

    env, get_action, get_value = load_policy(args.fpath, args.env, workload_file, args.itr if args.itr >=0 else 'last')
    run_policy(env, get_action, get_value, args.len, args.episodes, False)