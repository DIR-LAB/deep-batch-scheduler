import numpy as np
import tensorflow as tf
import gym
import time
import os
import json
from random import shuffle
import scipy.signal
import math

import hpc
from gym.spaces import Box, Discrete
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

MAX_QUEUE_SIZE = 64
MAX_JOBS_EACH_BATCH = 64
MIN_JOBS_EACH_BATCH = 64
MAX_MACHINE_SIZE = 256
MAX_WAIT_TIME = 12 * 60 * 60 # assume maximal wait time is 12 hours.
MAX_RUN_TIME = 12 * 60 * 60 # assume maximal runtime is 12 hours
SAMPLE_ACTIONS = 256

SEED = 42

# each job has three features: submit_time, request_number_of_processors, request_time/run_time,
JOB_FEATURES = 3

# ResNet HP
BN_EPSILON = 0.001
weight_decay = 0.0002  # scale for l2 regularization
num_residual_blocks = 5 # How many residual blocks do you want

# Exploration
explore_rate = 0.1


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))


def placeholders(*args):
    return [placeholder(dim) for dim in args]


def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError


def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]

# ResNet Architecture
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
        assert conv3.get_shape().as_list()[1:] == [10, 2, 64]

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

def resnet(x_ph, act_dim):
    '''
    https://github.com/wenxinxu/resnet-in-tensorflow
    '''
    sq = int(math.ceil(math.sqrt(MAX_QUEUE_SIZE)))
    job_queue_row = sq
    machine_row = int(math.ceil(MAX_MACHINE_SIZE / sq))

    x = tf.reshape(x_ph, shape=[-1, (job_queue_row + machine_row), sq, JOB_FEATURES])
    return inference(x, act_dim, num_residual_blocks, reuse=False)


# Basic CNN Architecture
def basic_cnn(x_ph, act_dim):
    sq = int(math.ceil(math.sqrt(MAX_QUEUE_SIZE)))
    job_queue_row = sq
    machine_row = int(math.ceil(MAX_MACHINE_SIZE / sq))
    print("Basic CNN, input dim:[%d, %d, %d]" % ((job_queue_row + machine_row), sq, JOB_FEATURES))

    x = tf.reshape(x_ph, shape=[-1, (job_queue_row + machine_row), sq, JOB_FEATURES])
    conv1 = tf.layers.conv2d(
            inputs=x,
            filters=32,
            kernel_size=[3, 3],
            strides=1,
            padding='same',
            activation=tf.nn.relu
    )
    pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2
    )
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[3, 3],
            strides=1,
            padding='same',
            activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=2
    )
    flat = tf.reshape(pool2, [-1, int((job_queue_row + machine_row)/4) * 2 * 64])
    dense = tf.layers.dense(
            inputs=flat,
            units=1024,
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


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]


def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def categorical_policy(x, a, action_space):
    act_dim = action_space.n
    # logits = resnet(x, act_dim)
    logits = basic_cnn(x, act_dim)
    # logits = mlp(x, list((256,256,256))+[act_dim], tf.tanh, None)
    logp_all = tf.nn.log_softmax(logits)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    # logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return logits, logp_all, logp


def actor_critic(x, a, action_space):
    with tf.variable_scope('pi'):
        logits, logp_all, logp = categorical_policy(x, a, action_space)
    with tf.variable_scope('v'):
        # v = tf.squeeze(resnet(x, 1), axis=1)
        v = tf.squeeze(basic_cnn(x, 1), axis=1)
        # v = tf.squeeze(mlp(x, list((256,256,256))+[1], tf.tanh, None), axis=1)
    return logits, logp_all, logp, v


class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a HPC VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
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
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf]

# pi_lr=0.001, vf_lr=1e-3,
def hpc_vpg(env_name, workload_file, model_path, ac_kwargs=dict(), seed=0,
            steps_per_epoch=4000, epochs=50, gamma=0.99,
            train_v_iters=20, lam=0.97, max_ep_len=10000,
            logger_kwargs=dict(), save_freq=10):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    mylr = 0.01
    lr_decay = 0.998

    env = gym.make(env_name)
    env.my_init(workload_file=workload_file)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph = placeholders_from_spaces(env.observation_space, env.action_space)
    adv_ph, ret_ph, logp_old_ph = placeholders(None, None, None)

    lr_ph = tf.placeholder(tf.float32, shape=None, name='learning_rate')

    # Main outputs from computation graph
    # pi, logp, logp_pi, v = cnn_actor_critic(x_ph, a_ph, **ac_kwargs)
    logits, logp_all, logp, v = actor_critic(x_ph, a_ph, **ac_kwargs)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [logits, logp_all, v]

    # Experience buffer
    buf = VPGBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    # Count variables
    var_counts = tuple(count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # VPG objectives
    pi_loss = -tf.reduce_mean(logp * adv_ph)
    v_loss = tf.reduce_mean((ret_ph - v) ** 2)

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp)  # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp)  # a sample estimate for entropy, also easy to compute

    # Optimizers
    train_pi = MpiAdamOptimizer(learning_rate=lr_ph).minimize(pi_loss)
    train_v = MpiAdamOptimizer(learning_rate=lr_ph).minimize(v_loss)

    # Loader Model from trained one: does not work as we have multiple models. (pi and v)
    # saver = tf.train.Saver()
    # saver.restore(sess, model_path)

    # Directly train it.
    loss_s = tf.losses.sparse_softmax_cross_entropy(labels=a_ph, logits=logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss_s, global_step=tf.train.get_global_step())

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    sample_json = []
    input_s = []
    label_s = []
    sample_cnt = 0

    print("loading SL training data:", model_path)
    with open(model_path, 'r') as f:
        for line in f:
            try:
                one_sample = json.loads(line)
                sample_json.append(one_sample)
            except:
                pass

    shuffle(sample_json)
    for sample in sample_json:
        input_s.append(sample['observe'])
        label_s.append(sample['label'])

    N_train = int(len(sample_json) * 0.9)
    sample_cnt = N_train
    feature_train = input_s[:N_train]
    feature_test = input_s[N_train:]
    label_train = label_s[:N_train]
    label_test = label_s[N_train:]

    index = 0
    batch_size = 200
    hm_epoch = 100

    def next_batch(index, batch_size):
        if index + batch_size > sample_cnt:
            x = np.array(feature_train[index:])
            y = np.array(label_train[index:])
            index = -1
        else:
            x = np.array(feature_train[index:index + batch_size])
            y = np.array(label_train[index:index + batch_size])
            index += batch_size
        return x, y, index

    for epoch in range(hm_epoch):
        epoch_loss = 0
        index = 0
        while index != -1:
            epoch_x, epoch_y, index = next_batch(index, batch_size)
            _, c = sess.run([train_op, loss_s], feed_dict={x_ph: epoch_x, a_ph: epoch_y})
            epoch_loss += c
        print('Epoch', epoch, 'completed out of', hm_epoch, 'loss:', epoch_loss)

    # Evaluation
    y_test = tf.placeholder(dtype=tf.int64, shape=(None,))
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    prediction = tf.argmax(pred, 1)
    correct_prediction = tf.equal(prediction, y_test)

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x_ph: feature_test, y_test: label_test}, session=sess))

    print("PreTrain the Model Finish")

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'v': v})  # 'pi': pi,

    def update(cur_lr):
        inputs = {k: v for k, v in zip(all_phs, buf.get())}
        inputs[lr_ph] = cur_lr

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

    def get_legal_action(interactions):
        rate = math.ceil((interactions + 1) / 100000) # reduce the exporation rate every 100K interactions.
        times = 0
        if np.random.rand() < (explore_rate / rate):
            while True:
                action = np.random.randint(0, MAX_QUEUE_SIZE)
                if action_is_legal(action):
                    return action, logp_all_t[0][action]
        else:
            while True:
                times += 1
                action = np.argmax(logits_t[0])
                if action_is_legal(action):
                    return action, logp_all_t[0][action]
                else:
                    logits_t[0][action] = (np.amin(logits_t[0]) - 1)
                # print("find action")
                if times > 64:
                    while True:
                        action = np.random.randint(0, MAX_QUEUE_SIZE)
                        if action_is_legal(action):
                            return action, logp_all_t[0][action]

    def action_is_legal(action):
        if all(o[0][action * JOB_FEATURES: (action + 1) * JOB_FEATURES] == 0):
            return False
        return True

    total_interactions = 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            logits_t, logp_all_t, v_t = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1, -1)})
            a, logp_t = get_legal_action(total_interactions)

            # save and log
            buf.store(o, a, r, v_t, logp_t)
            logger.store(VVals=v_t)

            o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1
            total_interactions += 1

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t == steps_per_epoch - 1):
                if not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = r if d else sess.run(v, feed_dict={x_ph: o.reshape(1, -1)})
                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)

        mylr *= lr_decay
        # Perform VPG update!
        update(mylr)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', total_interactions)
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
    parser.add_argument('--env', type=str, default='Scheduler-v1') # Scheduler-v1 is the job scheduler env.
    parser.add_argument('--workload', type=str, default='../../data/lublin_256.swf') # RICC-2010-2
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--exp_name', type=str, default='hpc-cnn-lubin-2000')
    args = parser.parse_args()

    mpi_fork(1)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs

    # build absolute path for using in hpc_env.
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    log_data_dir = os.path.join(current_dir, '../../data/logs/')
    logger_kwargs = setup_logger_kwargs(args.exp_name, seed=args.seed, data_dir=log_data_dir)

    hpc_vpg(args.env, workload_file, '../../data/lubin-SL-Shortest.txt', ac_kwargs=dict(), gamma=args.gamma,
            seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
            logger_kwargs=logger_kwargs)