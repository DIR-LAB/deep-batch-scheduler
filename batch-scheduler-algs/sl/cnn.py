import tensorflow as tf
import numpy as np

import json


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

    x_ph = tf.placeholder(dtype=tf.float32, shape=(None, 64))
    a_ph = tf.placeholder(dtype=tf.int32, shape=(None,))
    act_dim = 64
    logits = basic_cnn(x_ph)
    labels = tf.one_hot(a_ph, depth=act_dim)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    input = []
    label = []
    sample_cnt = 0
    training_samples = "../../data/RICC-SL-Shortest-JF-sorted.txt"
    with open(training_samples, 'r') as f:
        sample_json =  json.load(f)
        for sample in sample_json:
            input.append(sample[0:-1])
            label.append(sample[-1])
            sample_cnt += 1


    index = 0
    batch_size = 100
    hm_epoch = 10

    def next_batch(index, batch_size):
        x = np.array()
        y = np.array()
        if index + batch_size > sample_cnt:
            return None, None, -1
        for i in range(index, index + batch_size):
            x = np.insert(x, -1, values = np.array(input[i]), axis=0)
            y = np.insert(y, -1, values=np.array(label[i]), axis=0)
        return x, y

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epoch):
            epoch_loss = 0
            epoch_x, epoch_y, index = next_batch(index, batch_size)
            _, c = sess.run([optimizer, loss], feed_dict={x_ph: epoch_x, a_ph: epoch_y})
            epoch_loss += c
        print('Epoch', epoch, 'completed out of', hm_epoch, 'loss:', epoch_loss)