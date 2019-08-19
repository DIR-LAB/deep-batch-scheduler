import numpy as np
import tensorflow as tf
import scipy.signal
import os
import math
import json
import time
import sys
import random
from random import shuffle

from HPCSimPickAlgms import *

from spinup.utils.logx import EpochLogger
from spinup.utils.logx import restore_tf_graph

np.set_printoptions(threshold=np.inf)

def mlp(x, act_dim):
    for _ in range(3):
        x = tf.layers.dense(x, units=MLP_SIZE, activation=tf.tanh)
    return tf.layers.dense(x, units=act_dim, activation=tf.tanh)

def basic_cnn(x_ph, act_dim):
    x = tf.reshape(x_ph, shape=[-1, 6, 6, JOB_FEATURES])
    conv1 = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=[1, 1],
            strides=1,
            activation=tf.nn.relu
    ) # 6 * 6
    pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=1
    ) # 5 * 5
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=32,
            kernel_size=[2, 2],
            strides=1,
            activation=tf.nn.relu
    ) # 4 * 4
    pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=2
    ) # 2 * 2
    flat = tf.reshape(pool2, [-1, 2 * 2 * 32])
    dense = tf.layers.dense(
            inputs=flat,
            units=256,
            activation=tf.nn.relu
    )
    dropout = tf.layers.dropout(
            inputs=dense,
            rate=0.5,
    )
    return tf.layers.dense(
            inputs=dropout,
            units=act_dim
    )

def dnn(x_ph, act_dim):
    x = tf.reshape(x_ph, shape=[-1, 6, 6, JOB_FEATURES])
    conv1 = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=[1, 1],
            strides=1,
            activation=tf.nn.relu
    ) # 6 * 6
    pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[6, 6],
            strides=1
    ) # 1 * 1
    flat = tf.reshape(pool1, [-1, 1 * 1 * 64])
    dense = tf.layers.dense(
            inputs=flat,
            units=256,
            activation=tf.nn.relu
    )
    return tf.layers.dense(
            inputs=dense,
            units=act_dim,
            activation=None
    )

"""
Policies
"""
def categorical_policy(x, a, action_space):
    act_dim = action_space.n
    output_layer = basic_cnn(x, act_dim)
    action_probs = tf.squeeze(tf.nn.softmax(output_layer))
    log_picked_action_prob = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * tf.nn.log_softmax(output_layer), axis=1)    
    return action_probs, log_picked_action_prob

"""
Actor-Critics
"""
def actor_critic(x, a, action_space=None):
    with tf.variable_scope('pi'):
        action_probs, log_picked_action_prob = categorical_policy(x, a, action_space)
    with tf.variable_scope('v'):
        v = tf.squeeze(basic_cnn(x, 1), axis=1)
    return action_probs, log_picked_action_prob, v

class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a HPC VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        size = size * 100 # assume the traj can be really long 
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
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
        # print("call finish path vith value", last_val)
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
        assert self.ptr < self.max_size
        actual_size = self.ptr
        self.ptr, self.path_start_idx = 0, 0

        actual_adv_buf = np.array(self.adv_buf, dtype = np.float32)
        actual_adv_buf = actual_adv_buf[:actual_size]
        # print ("-----------------------> actual_adv_buf: ", actual_adv_buf)
        adv_sum = np.sum(actual_adv_buf)
        adv_n = len(actual_adv_buf)
        adv_mean = adv_sum / adv_n
        adv_sum_sq = np.sum((actual_adv_buf - adv_mean) ** 2)
        adv_std = np.sqrt(adv_sum_sq / adv_n)
        # print ("-----------------------> adv_std:", adv_std)
        actual_adv_buf = (actual_adv_buf - adv_mean) / adv_std
        # print (actual_adv_buf)

        return [self.obs_buf[:actual_size], self.act_buf[:actual_size], actual_adv_buf,
                self.ret_buf[:actual_size], self.logp_buf[:actual_size]]

def vpg(workload_file, model_path, ac_kwargs=dict(), seed=0,
        traj_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000000,
        logger_kwargs=dict(), save_freq=10, pre_trained = 0, trained_model_path = ''):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    env = HPCEnv()
    env.my_init(workload_file=workload_file, sched_file=model_path)
    env.seed(seed)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    ac_kwargs['action_space'] = env.action_space

    # x_ph has shape=(None, *(MAX_QUEUE_SIZE * JOB_FEATURES))
    # a_ph, adv_ph, ret_ph, log_old_ph have shape = (None, )
    x_ph, a_ph = placeholders_from_spaces(env.observation_space, env.action_space)
    adv_ph, ret_ph, logp_old_ph = placeholders(None, None, None)
    
    # Main outputs from computation graph
    action_probs, log_picked_action_prob, v = actor_critic(x_ph, a_ph, **ac_kwargs)
    
    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]
    
    # Every step, get: action, value
    get_action_ops = [action_probs, v]

    # Experience buffer
    buf = VPGBuffer(obs_dim, act_dim, traj_per_epoch * JOB_SEQUENCE_SIZE, gamma, lam)

    # Count variables
    var_counts = tuple(count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # vpg objectives
    pi_loss = -tf.reduce_mean(log_picked_action_prob * adv_ph)
    v_loss = tf.reduce_mean((ret_ph - v) ** 2)
    
    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - log_picked_action_prob)  # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-log_picked_action_prob)  # a sample estimate for entropy, also easy to compute
    
    # optimizers
    train_pi = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    train_v = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Setup model saving
    # logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'action_probs': action_probs, 'log_picked_action_prob': log_picked_action_prob, 'v': v})
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a':a_ph, 'adv':adv_ph, 'ret':ret_ph, 'logp_old_ph':logp_old_ph}, outputs={'action_probs': action_probs, 'log_picked_action_prob': log_picked_action_prob, 'v': v, 'pi_loss':pi_loss, 'v_loss':v_loss, 'approx_ent':approx_ent, 'approx_kl':approx_kl})

    def update():
        inputs = {k: v for k, v in zip(all_phs, buf.get())}
        pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)

        # Policy gradient step
        sess.run(train_pi, feed_dict=inputs)

        # Value function learning
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl = sess.run([pi_loss, v_loss, approx_kl], feed_dict=inputs)
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        t = 0
        while True:
            # x_ph should be (?, 144)
            action_probs, v_t = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES)})
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            log_action_prob = np.log(action_probs[action])
            
            # save and log
            buf.store(o, np.array(action), r, v_t, log_action_prob)
            logger.store(VVals=v_t)

            o, r, d, _ = env.step(action)
            ep_ret += r
            ep_len += 1

            if d:
                t += 1
                buf.finish_path(r)
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                if t >= traj_per_epoch:
                    # print ("state:", state, "\nlast action in a traj: action_probs:\n", action_probs, "\naction:", action)
                    break
                
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)

        # Perform VPG update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', with_min_and_max=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * traj_per_epoch * JOB_SEQUENCE_SIZE)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default='./data/lublin_256.swf')  # RICC-2010-2
    parser.add_argument('--model', type=str, default='./data/lublin_256.schd')
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--trajs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--exp_name', type=str, default='vpg-pick-algms')
    parser.add_argument('--pre_trained', type=int, default=0)
    parser.add_argument('--trained_model', type=str, default='./data/logs/reinforce-model/reinforce-s0/')

    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    # build absolute path for using in hpc_env.
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    log_data_dir = os.path.join(current_dir, './data/logs/')
    logger_kwargs = setup_logger_kwargs(args.exp_name, seed=args.seed, data_dir=log_data_dir)

    vpg(workload_file, args.model, gamma=args.gamma,
        seed=args.seed, traj_per_epoch=args.trajs, epochs=args.epochs,
        logger_kwargs=logger_kwargs, pre_trained = args.pre_trained, trained_model_path=args.trained_model)
