import time
import joblib
import os
import os.path as osp
import tensorflow as tf
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph

import gym
from gym import spaces
from gym.spaces import Box, Discrete
from gym.utils import seeding

import random
import math
import numpy as np
import sys

from HPCSimPickJobs import *

import matplotlib.pyplot as plt
plt.rcdefaults()

def load_policy(model_path, itr='last'):
    # handle which epoch to load from
    if itr=='last':
        saves = [int(x[11:]) for x in os.listdir(model_path) if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(model_path, 'simple_save'+itr))

    # get the correct op for executing actions
    action_probs = model['action_probs']
    v = model['v']

    # make function for producing an action given a single state
    get_probs = lambda x : sess.run(action_probs, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES)})
    get_v = lambda x : sess.run(v, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES)})
    return get_probs, get_v

def run_policy(env, get_probs, get_value, nums, iters):
    rl_r = []
    f1_r = [] 
    f2_r = []
    f3_r = []
    f4_r = []
    sjf_r = []
    small_r = []
    fcfs_r = []

    for _ in range(0, iters):
        env.reset_for_test(nums)
        f1_r.append(sum(env.schedule_curr_sequence_reset(env.f1_score).values()))
        f2_r.append(sum(env.schedule_curr_sequence_reset(env.f2_score).values()))
        f3_r.append(sum(env.schedule_curr_sequence_reset(env.f3_score).values()))
        f4_r.append(sum(env.schedule_curr_sequence_reset(env.f4_score).values()))
        
        sjf_r.append(sum(env.schedule_curr_sequence_reset(env.sjf_score).values()))
        small_r.append(sum(env.schedule_curr_sequence_reset(env.smallest_score).values()))
        fcfs_r.append(sum(env.schedule_curr_sequence_reset(env.fcfs_score).values()))

        o = env.build_observation()
        print ("schedule: ", end="")
        rl = 0
        while True:
            action_probs = get_probs(o)
            # v_t = get_value(o)

            lst = []
            legal_job_idx = []
            for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
                if o[i] == 0 and o[i+1] == 1 and o[i+2] == 1 and o[i+3] == 0:
                    lst.append(0)
                elif o[i] == 1 and o[i+1] == 1 and o[i+2] == 1 and o[i+3] == 1:
                    lst.append(0)
                else:
                    lst.append(action_probs[int(i/JOB_FEATURES)])
                    legal_job_idx.append(int(i/JOB_FEATURES))

            legal_action_probs = np.array(lst)
            total = legal_action_probs.sum()
            legal_action_probs /= total

            if np.isnan(legal_action_probs).any():
                print("nan:---------->observation:\n", o, "\nlegal_action_probs", legal_action_probs, "\naction_probs", action_probs)
                return

            # a = np.argmax(legal_action_probs)
            a = np.random.choice(np.arange(MAX_QUEUE_SIZE), p=legal_action_probs)
            print (a, end=" ")
            o, r, d, _ = env.step_for_test(a)
            rl += r
            if d:
                break
        rl_r.append(rl)
        print ("")

    # plot
    all_data = []
    all_data.append(rl_r)
    all_data.append(sjf_r)
    all_data.append(small_r)
    #all_data.append(fcfs_r)
    
    all_data.append(f1_r)
    all_data.append(f2_r)
    all_data.append(f3_r)
    all_data.append(f4_r)

    all_medians = []
    for p in all_data:
        all_medians.append(np.median(p))

    # plt.rc("font", size=35)
    plt.figure(figsize=(16, 14))
    axes = plt.axes()

    xticks = [y + 1 for y in range(len(all_data))]
    plt.plot(xticks[0:1], all_data[0:1], 'o', color='darkorange')
    plt.plot(xticks[1:2], all_data[1:2], 'o', color='darkorange')
    plt.plot(xticks[2:3], all_data[2:3], 'o', color='darkorange')
    plt.plot(xticks[3:4], all_data[3:4], 'o', color='darkorange')
    plt.plot(xticks[4:5], all_data[4:5], 'o', color='darkorange')
    plt.plot(xticks[5:6], all_data[5:6], 'o', color='darkorange')
    plt.plot(xticks[6:7], all_data[6:7], 'o', color='darkorange')

    plt.boxplot(all_data, showfliers=False)

    axes.yaxis.grid(True)
    axes.set_xticks([y + 1 for y in range(len(all_data))])
    xticklabels = ['RL', 'SJF', 'SMALL', 'F1', 'F2', 'F3', 'F4']
    plt.setp(axes, xticks=[y + 1 for y in range(len(all_data))],
             xticklabels=xticklabels)

    plt.tick_params(axis='both', which='major', labelsize=35)
    plt.tick_params(axis='both', which='minor', labelsize=35)

    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rlmodel', type=str)
    parser.add_argument('--workload', type=str, default='./data/lublin_256.swf')
    parser.add_argument('--len', '-l', type=int, default=128)
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--iter', '-i', type=int, default=10)
    args = parser.parse_args()

    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    model_file = os.path.join(current_dir, args.rlmodel)

    get_probs, get_value = load_policy(model_file, 'last') 
    
    # initialize the environment from scratch
    env = HPCEnv()
    env.my_init(workload_file=workload_file)
    env.seed(int(time.time()))

    run_policy(env, get_probs, get_value, args.len, args.iter)