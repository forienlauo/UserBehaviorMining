# coding=utf-8
import os
import logging

import time

from util.sampler import random_sample

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np


def cnn(argv):
    # argv
    train_data = np.loadtxt(argv[1])
    test_data = np.loadtxt(argv[2])

    # configuration
    initial_height, initial_width = int(argv[3]), int(argv[4])
    initial_in_chanels = 1
    target_class_cnt = int(argv[5])

    CONV_STRIDES_H, CONV_STRIDES_W = 1, 1
    CONV_HEIGHT, CONV_WIDTH = 5, 5

    POOL_STRIDES_H, POOL_STRIDES_W = 2, 2
    POOL_SHAPE = [1, 2, 2, 1]

    KEEP_PROB = 0.5

    ITERATION = 1000
    BATCH_SIZE = 50

    sess = tf.InteractiveSession()

    # input and labels
    x = tf.placeholder(tf.float32, shape=[None, initial_height * initial_width])
    y_ = tf.placeholder(tf.float32, shape=[None, target_class_cnt])

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W, strides=(1, CONV_STRIDES_H, CONV_STRIDES_W, 1), padding='SAME'):
        # 如果使用默认步长strides[height, width]=[1,1], 则卷积后的图片大小不变
        return tf.nn.conv2d(x, W, strides, padding)

    def max_pool(x, ksize, strides=(1, POOL_STRIDES_H, POOL_STRIDES_W, 1), padding='SAME'):
        # 如果使用默认步长strides[height, width]=[2,2], 则卷积后的图片大小变为原来的一半
        return tf.nn.max_pool(x, ksize, strides, padding)

    class Utils(object):
        @staticmethod
        def format_inputs(example):
            example = np.array(example)
            return example[:, :-target_class_cnt], example[:, -target_class_cnt:]

    # First Convolutional Layer
    _in_channels1 = initial_in_chanels
    _out_channels1 = 32
    _example_cnt, _height1, _width1 = -1, initial_height, initial_width
    x_image = tf.reshape(x, [_example_cnt, _height1, _width1, _in_channels1])
    W_conv1 = weight_variable([CONV_HEIGHT, CONV_WIDTH, _in_channels1, _out_channels1])
    b_conv1 = bias_variable([_out_channels1])

    in1 = x_image  # shape[_example_cnt, _height1, _width1, _in_channels1]
    h_conv1 = tf.nn.relu(conv2d(in1, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1, POOL_SHAPE)
    out1 = h_pool1  # shape[_example_cnt, _height1 / 2, _width1 / 2, _out_channels1]

    # # Second Convolutional Layer
    # _in_channels2 = _out_channels1
    # _out_channels2 = 64
    # _example_cnt, _height2, _width2 = -1, _height1 / POOL_STRIDES_H, _width1 / POOL_STRIDES_W
    # W_conv2 = weight_variable([CONV_HEIGHT, CONV_WIDTH, _in_channels2, _out_channels2])
    # b_conv2 = bias_variable([_out_channels2])
    #
    # in2 = out1  # shape[_example_cnt, _height2, _width2, _in_channels2]
    # h_conv2 = tf.nn.relu(conv2d(in2, W_conv2) + b_conv2)
    # h_pool2 = max_pool(h_conv2, POOL_SHAPE)
    # out2 = h_pool2  # shape[_example_cnt, _height2 / 2, _width2 / 2, _out_channels2]
    _out_channels2 = _out_channels1
    _height2 = _height1
    _width2 = _width1
    out2 = out1

    # Densely Connected Layer(Full Connected Layer)
    _in_channels3 = _out_channels2
    _out_channels3 = 1024
    _example_cnt, _height3, _width3 = -1, _height2 / POOL_STRIDES_H, _width2 / POOL_STRIDES_W
    W_fc1 = weight_variable([_height3 * _width3 * _in_channels3, _out_channels3])
    b_fc1 = bias_variable([_out_channels3])

    in3 = out2  # shape[_example_cnt, _height3, _width3, _in_channels3]
    h_pool2_flat = tf.reshape(in3, [_example_cnt, _height3 * _width3 * _in_channels3])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 这不再卷积,直接矩阵乘
    _out_width3 = _out_channels3
    out3 = h_fc1  # shape[_example_cnt, _out_width3]

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    tmp_in = out3
    h_fc1_drop = tf.nn.dropout(tmp_in, keep_prob)
    tmp_out = h_fc1_drop
    out3 = tmp_out

    # Readout Layer
    _in_width4 = _out_width3
    _out_width4 = target_class_cnt
    _example_cnt = -1
    W_fc2 = weight_variable([_in_width4, _out_width4])
    b_fc2 = bias_variable([_out_width4])

    in4 = out3  # shape[_example_cnt, _in_width4]
    y_conv = tf.matmul(in4, W_fc2) + b_fc2
    out4 = y_conv  # shape[_example_cnt, _out_width4]

    # Train
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_per_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    logging.info("start to train cnn.")
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    for i in range(ITERATION):
        batch = random_sample(train_data, BATCH_SIZE)
        # print progress
        _X, _Y = Utils.format_inputs(batch)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: _X, y_: _Y, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_per_step.run(feed_dict={x: _X, y_: _Y, keep_prob: KEEP_PROB})
    end_time = time.time()
    logging.info("end to train cnn.")
    logging.info('cost time: %.2fs' % (end_time - start_time))

    # Evaluate
    _X, _Y = Utils.format_inputs(test_data)
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: _X, y_: _Y, keep_prob: 1.0}))

    sess.close()
