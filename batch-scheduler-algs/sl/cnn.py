import tensorflow as tf
import numpy as np
from random import shuffle

import csv
import json
import sys
import os

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

def resnet(x_ph):
    '''
    https://github.com/wenxinxu/resnet-in-tensorflow
    '''
    x = tf.reshape(x_ph, shape=[-1, 8, 8, 3])
    return inference(x, 64, num_residual_blocks, reuse=False)


def basic_cnn(x_ph):
    x = tf.reshape(x_ph, shape=[-1, 8, 8, 3])
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
    flat = tf.reshape(pool2, [-1, 2 * 2 * 64])
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
            units=64
    )


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: python cnn.py training-data-path ouput-dir")
        sys.exit()

    x_ph = tf.placeholder(dtype=tf.float32, shape=(None, 192))
    a_ph = tf.placeholder(dtype=tf.int32, shape=(None,))
    act_dim = 64
    logits = basic_cnn(x_ph)
    # labels = tf.one_hot(a_ph, depth=act_dim)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=a_ph, logits=logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    input = []
    label = []
    sample_cnt = 0
    # "../../data/RICC-SL-Shortest-JF-sorted.txt"
    training_samples = sys.argv[1]

    sample_json = []
    with open(training_samples, 'r') as f:
        for line in f:
            try:
                one_sample = json.loads(line)
                sample_json.append(one_sample)
            except:
                pass

    # print (len(sample_json))
    shuffle(sample_json)
    for sample in sample_json:
        input.append(sample['observe'])
        label.append(sample['label'])

    N_train = int(len(sample_json) * 0.8)
    sample_cnt = N_train
    feature_train = input[:N_train]
    feature_test = input[N_train:]
    label_train = label[:N_train]
    label_test = label[N_train:]

    index = 0
    batch_size = 100
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


    # x, y, index = next_batch(index, batch_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epoch):
            epoch_loss = 0
            index = 0
            while index != -1:
                epoch_x, epoch_y, index = next_batch(index, batch_size)
                # print (len(epoch_x), len(epoch_y))
                _, c = sess.run([train_op, loss], feed_dict={x_ph: epoch_x, a_ph: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epoch, 'loss:', epoch_loss)

        # Evaluation
        y_test = tf.placeholder(dtype=tf.int64, shape=(None,))
        pred = tf.nn.softmax(logits)  # Apply softmax to logits
        prediction = tf.argmax(pred, 1)
        correct_prediction = tf.equal(prediction, y_test)

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x_ph: feature_test, y_test: label_test}))
        pred_save = prediction.eval({x_ph: feature_test, y_test: label_test})

    output_dir = sys.argv[2]
    label_file = os.path.join(output_dir, "label.csv")
    predict_file = os.path.join(output_dir, "predict.csv")
    with open(label_file, 'w') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(label_test)
    with open(predict_file, 'w') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(pred_save.tolist())