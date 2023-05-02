import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from HPCSimPickJobs import *
import os.path as osp
import numpy as np
import time
import math
import os


plt.rcdefaults()

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

#@profile
def run_policy(env, get_probs, nums, iters, score_type):
    rl_r = []
    f1_r = [] 
    f2_r = []
    sjf_r = []
    wfp_r = []
    uni_r = []

    fcfs_r = []

    # time_total = 0
    # num_total = 0
    for iter_num in range(0, iters):
        start = iter_num *args.len
        env.reset_for_test(nums,start)
        f1_r.append(sum(env.schedule_curr_sequence_reset(env.f1_score).values()))
        # f2_r.append(sum(env.schedule_curr_sequence_reset(env.f2_score).values()))
        uni_r.append(sum(env.schedule_curr_sequence_reset(env.uni_score).values()))
        wfp_r.append(sum(env.schedule_curr_sequence_reset(env.wfp_score).values()))
        
        sjf_r.append(sum(env.schedule_curr_sequence_reset(env.sjf_score).values()))
        #small_r.append(sum(env.schedule_curr_sequence_reset(env.smallest_score).values()))
        fcfs_r.append(sum(env.schedule_curr_sequence_reset(env.fcfs_score).values()))

        o = env.build_observation()
        print ("schedule: ", end="")
        rl = 0
        rl_decisions = 0
        while True:
            count = 0
            skip_ = []
            lst = []
            for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
                if all(o[i:i + JOB_FEATURES] == [0] + [1] * (JOB_FEATURES - 2) + [0]):
                    lst.append(0)
                elif all(o[i:i + JOB_FEATURES] == [1] * JOB_FEATURES):
                    lst.append(0)
                else:
                    count += 1
                    if all(o[i:i + JOB_FEATURES] == [1] * (JOB_FEATURES-1) + [0]):
                        skip_.append(math.floor(i/JOB_FEATURES))
                    lst.append(1)
            pi = get_probs(o)
            a = pi[0]
            rl_decisions += 1.0

            if a in skip_:
                print("SKIP" + "(" + str(count) + ")", end="|")
            else:
                print (str(a)+"("+str(count)+")", end="|")
            o, r, d, _ = env.step_for_test(a)
            rl += r
            if d:
                # print("RL decision ratio:",rl_decisions/total_decisions)
                print("Sequence Length:",rl_decisions)
                break
        rl_r.append(rl)
        print ("")

    # plot
    all_data = []
    all_data.append(fcfs_r)
    all_data.append(wfp_r)
    all_data.append(uni_r)
    all_data.append(sjf_r)
    all_data.append(f1_r)
    all_data.append(rl_r)
    #all_data.append(fcfs_r)
    

    all_medians = []
    for p in all_data:
        all_medians.append(np.median(p))

    # plt.rc("font", size=45)
    # plt.figure(figsize=(12, 7))
    # plt.ylim([0,400])
    plt.rc("font", size=23)
    plt.figure(figsize=(9, 5))
    axes = plt.axes()

    xticks = [y + 1 for y in range(len(all_data))]
    plt.plot(xticks[0:1], all_data[0:1], 'o', color='darkorange')
    plt.plot(xticks[1:2], all_data[1:2], 'o', color='darkorange')
    plt.plot(xticks[2:3], all_data[2:3], 'o', color='darkorange')
    plt.plot(xticks[3:4], all_data[3:4], 'o', color='darkorange')
    plt.plot(xticks[4:5], all_data[4:5], 'o', color='darkorange')
    plt.plot(xticks[5:6], all_data[5:6], 'o', color='darkorange')
    #plt.plot(xticks[6:7], all_data[6:7], 'o', color='darkorange')

    plt.boxplot(all_data, showfliers=False, meanline=True, showmeans=True, medianprops={"linewidth":0},meanprops={"color":"darkorange", "linewidth":4,"linestyle":"solid"})


    axes.yaxis.grid(True)
    axes.set_xticks([y + 1 for y in range(len(all_data))])
    xticklabels = ['FCFS', 'WFP', 'UNI', 'SJF', 'F1', 'RL']
    # xticklabels = ['FCFS', 'WFP', 'UNI', 'SJF', 'RL']
    plt.setp(axes, xticks=[y + 1 for y in range(len(all_data))],
             xticklabels=xticklabels)
    if score_type == 0:
        plt.ylabel("Average bounded slowdown")
    elif score_type == 1:
        plt.ylabel("Average waiting time")
    elif score_type == 2:
        plt.ylabel("Average turnaround time")
    elif score_type == 3:
        plt.ylabel("Resource utilization")
    else:
        raise NotImplementedError
    
    plt.xlabel("Scheduling Policies")
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)

    plt.show()

if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--rlmodel', type=str, default="./data/ppo_HPC")
    parser.add_argument('--workload', type=str, default='./data/lublin_256.swf')
    parser.add_argument('--len', '-l', type=int, default=1024)
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--iter', '-i', type=int, default=10)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--backfil', type=int, default=1)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--score_type', type=int, default=0)
    parser.add_argument('--batch_job_slice', type=int, default=0)

    args = parser.parse_args()

    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    model_file = os.path.join(current_dir, args.rlmodel)

    # initialize the environment from scratch
    env = HPCEnv(shuffle=args.shuffle, backfil=args.backfil, skip=args.skip, job_score_type=args.score_type,
                 batch_job_slice=args.batch_job_slice, build_sjf=False)
    env.my_init(workload_file=workload_file)
    env.seed(args.seed) 

    model = PPO.load(args.rlmodel, env=env)
    
    

    start = time.time()

    run_policy(env, model.predict, args.len, args.iter, args.score_type)
    print("elapse: {}".format(time.time()-start))