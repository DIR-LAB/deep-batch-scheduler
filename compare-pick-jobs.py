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
tf.enable_eager_execution()

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
    pi = model['pi']
    v = model['v']
    out = model['out']
    get_out = lambda x ,y  : sess.run(out, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES), model['mask']:y.reshape(-1, MAX_QUEUE_SIZE)})
    # make function for producing an action given a single state
    get_probs = lambda x ,y  : sess.run(pi, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES), model['mask']:y.reshape(-1, MAX_QUEUE_SIZE)})
    get_v = lambda x : sess.run(v, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES)})
    return get_probs, get_out

def action_from_obs(o):
    lst = []
    for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
        if o[i] == 0 and o[i + 1] == 1 and o[i + 2] == 1 and o[i + 3] == 0:
            pass
        elif o[i] == 1 and o[i + 1] == 1 and o[i + 2] == 1 and o[i + 3] == 1:
            pass
        else:
            lst.append((o[i+1],math.floor(i/JOB_FEATURES)))
    min_time = min([i[0] for i in lst])
    result = [i[1] for i in lst if i[0]==min_time]
    return result[0]
def run_policy(env, get_probs, get_out, nums, iters):
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
        total_decisions = 0
        rl_decisions = 0
        while True:
            count = 0
            skip_ = []
            lst = []
            for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
                if o[i] == 0 and o[i+1] == 1 and o[i+2] == 1 and o[i+3] == 0:
                    lst.append(0)
                elif o[i] == 1 and o[i+1] == 1 and o[i+2] == 1 and o[i+3] == 1:
                    lst.append(0)
                else:
                    count += 1
                    if o[i] == 1 and o[i+1] == 1 and o[i+2] == 1 and o[i+3] == 0:
                        skip_.append(math.floor(i/JOB_FEATURES))
                    lst.append(1)
            out = get_out(o,np.array(lst))
            softmax_out = tf.nn.softmax(out)
            confidence = tf.reduce_max(softmax_out)
            total_decisions += 1.0
            if confidence > 0.8:
                pi = get_probs(o, np.array(lst))
                a = pi[0]
                rl_decisions += 1.0
            else:
                # print('SJF')
                a = action_from_obs(o)
            # print(out)
            # v_t = get_value(o)



            if a in skip_:
                print("SKIP" + "(" + str(count) + ")", end="|")
            else:
                print (str(a)+"("+str(count)+")", end="|")
            o, r, d, _ = env.step_for_test(a)
            rl += r
            if d:
                print("RL decision ratio:",rl_decisions/total_decisions)
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

    plt.boxplot(all_data, showfliers=False, meanline=True, showmeans=True)

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
    parser.add_argument('--rlmodel', type=str, default="/home/dzhang16/zhangdi/projects/f1_skip/ppo_job_newmask/ppo_job_newmask_s0/")
    parser.add_argument('--workload', type=str, default='/home/dzhang16/zhangdi/projects/deep-batch-scheduler/data/SDSC-BLUE-2000-4.2-cln.swf')
    parser.add_argument('--len', '-l', type=int, default=1024)
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--iter', '-i', type=int, default=20)
    args = parser.parse_args()

    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    model_file = os.path.join(current_dir, args.rlmodel)

    get_probs, get_value = load_policy(model_file, 'last') 
    
    # initialize the environment from scratch
    env = HPCEnv()
    env.my_init(workload_file=workload_file)
    env.seed(args.seed)

    run_policy(env, get_probs, get_value, args.len, args.iter)
