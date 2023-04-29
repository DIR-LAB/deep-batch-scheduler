from spinup.utils.logx import restore_tf_graph
from tensorflow.python.util import deprecation
from HPCSimPickJobs import *
import tensorflow as tf
import os.path as osp
import logging
import time
import os

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True
tf.enable_eager_execution()


def load_policy(model_path, itr='last'):
    # handle which epoch to load from
    if itr == 'last':
        saves = [int(x[11:]) for x in os.listdir(model_path) if 'simple_save' in x and len(x) > 11]
        itr = '%d' % max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d' % itr

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(model_path, 'simple_save' + itr))

    # get the correct op for executing actions
    pi = model['pi']
    v = model['v']
    out = model['out']
    get_out = lambda x, y: sess.run(out, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES),
                                                    model['mask']: y.reshape(-1, MAX_QUEUE_SIZE)})
    # make function for producing an action given a single state
    get_probs = lambda x, y: sess.run(pi, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES),
                                                     model['mask']: y.reshape(-1, MAX_QUEUE_SIZE)})
    get_v = lambda x: sess.run(v, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES)})
    return get_probs, get_out


def action_from_obs(o):
    lst = []
    for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
        if o[i] == 0 and o[i + 1] == 1 and o[i + 2] == 1 and o[i + 3] == 0:
            pass
        elif o[i] == 1 and o[i + 1] == 1 and o[i + 2] == 1 and o[i + 3] == 1:
            pass
        else:
            lst.append((o[i + 1], math.floor(i / JOB_FEATURES)))
    min_time = min([i[0] for i in lst])
    result = [i[1] for i in lst if i[0] == min_time]
    return result[0]


# @profile
def run_policy(env, get_probs, get_out, nums, iters, score_type):
    rl_r = []
    f1_r = []
    f2_r = []
    sjf_r = []
    # small_r = []
    wfp_r = []
    uni_r = []

    fcfs_r = []

    # time_total = 0
    # num_total = 0
    for iter_num in range(0, iters):
        start = iter_num * args.len
        env.reset_for_test(nums, start)
        f1_r.append(sum(env.schedule_curr_sequence_reset(env.f1_score).values()))
        # f2_r.append(sum(env.schedule_curr_sequence_reset(env.f2_score).values()))
        uni_r.append(sum(env.schedule_curr_sequence_reset(env.uni_score).values()))
        wfp_r.append(sum(env.schedule_curr_sequence_reset(env.wfp_score).values()))

        sjf_r.append(sum(env.schedule_curr_sequence_reset(env.sjf_score).values()))
        # small_r.append(sum(env.schedule_curr_sequence_reset(env.smallest_score).values()))
        fcfs_r.append(sum(env.schedule_curr_sequence_reset(env.fcfs_score).values()))

        o = env.build_observation()
        rl = 0
        total_decisions = 0
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
                    if all(o[i:i + JOB_FEATURES] == [1] * (JOB_FEATURES - 1) + [0]):
                        skip_.append(math.floor(i / JOB_FEATURES))
                    lst.append(1)

            out = get_out(o, np.array(lst))
            softmax_out = tf.nn.softmax(out)
            confidence = tf.reduce_max(softmax_out)
            total_decisions += 1.0
            if confidence > 0:
                # start_time = time.time()
                pi = get_probs(o, np.array(lst))
                # pi = tf.arg_max(softmax_out, dimension=1)
                # time_total += time.time() - start_time
                # num_total += 1
                # print(start_time, time_total, num_total)
                a = pi[0]
                rl_decisions += 1.0
            else:
                # print('SJF')
                a = action_from_obs(o)
            # print(out)
            # v_t = get_value(o)


            o, r, d, _ = env.step_for_test(a)
            rl += r
            if d:
                # print("RL decision ratio:",rl_decisions/total_decisions)
                break
        rl_r.append(rl)

    # plot
    all_data = []
    all_data.append(fcfs_r)
    all_data.append(wfp_r)
    all_data.append(uni_r)
    all_data.append(sjf_r)
    all_data.append(f1_r)
    all_data.append(rl_r)
    # all_data.append(fcfs_r)

    all_medians = []
    for p in all_data:
        all_medians.append(np.median(p))
    all_means = []
    for p in all_data:
        all_means.append(np.mean(p))
    print(*all_means)



if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--rlmodel', type=str, default="./trained_models/sdsc_sp2/sdsc_sp2_s4")
    parser.add_argument('--workload', type=str, default='./data/lublin_256.swf')
    parser.add_argument('--len', '-l', type=int, default=10)
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--iter', '-i', type=int, default=10)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--backfil', type=int, default=0)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--score_type', type=int, default=0)
    parser.add_argument('--batch_job_slice', type=int, default=0)

    args = parser.parse_args()

    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    model_file = os.path.join(current_dir, args.rlmodel)

    get_probs, get_value = load_policy(model_file, 'last')

    # initialize the environment from scratch
    env = HPCEnv(shuffle=args.shuffle, backfil=args.backfil, skip=args.skip, job_score_type=args.score_type,
                 batch_job_slice=args.batch_job_slice, build_sjf=False)
    env.my_init(workload_file=workload_file)
    env.seed(args.seed)

    start = time.time()
    run_policy(env, get_probs, get_value, args.len, args.iter, args.score_type)
    print("elapse: {}".format(time.time() - start))