import tensorflow as tf
import numpy as np
from random import shuffle

import csv
import json
import sys
import os


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