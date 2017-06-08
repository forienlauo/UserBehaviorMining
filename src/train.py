# coding=utf-8
import logging
import os
import time

import numpy as np
import tensorflow as tf

from src.util.sampler import random_sample


class CNNTrainer(object):
    # cnn configuration
    CONV_STRIDES_H, CONV_STRIDES_W = 1, 1
    CONV_HEIGHT, CONV_WIDTH = 5, 5

    POOL_STRIDES_H, POOL_STRIDES_W = 2, 2
    POOL_SHAPE = [1, 2, 2, 1]

    KEEP_PROB = 0.5

    @staticmethod
    def dump(sess, model_file_path, ):
        """保存一个sess中的全部变量
        实际并不存在文件路径 model_file_path, dirname(model_file_path) 作为保存的目录, basename(model_file_path) 作为模型的名字
        @:return 模型保存保存的目录,即 dirname(model_file_path)
        """
        saver = tf.train.Saver()
        saver.save(sess, model_file_path)

        model_dir = os.path.dirname(model_file_path)
        return model_dir

    @staticmethod
    def load(sess, model_file_path, ):
        """加载一个sess中的全部变量
        实际并不存在文件路径 model_file_path, dirname(model_file_path) 作为保存的目录, basename(model_file_path) 作为模型的名字
        @:return sess中的 gragh, 可以通过 graph 取得所有 tensor 和 operation
        """
        meta_gragh_path = '%s.meta' % (model_file_path,)
        saver = tf.train.import_meta_graph(meta_gragh_path)
        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(model_file_path)))

        return tf.get_default_graph()

    @staticmethod
    def __format_inputs(example, target_class_cnt):
        example = np.array(example)
        return example[:, :-target_class_cnt], example[:, -target_class_cnt:]

    @staticmethod
    def construct(
            initial_height, initial_width, initial_channels, target_class_cnt,
            x, y_, keep_prob,
    ):
        def weight_variable(shape, name=None, ):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name, )

        def bias_variable(shape, name=None, ):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name, )

        def conv2d(x, W,
                   strides=(1, CNNTrainer.CONV_STRIDES_H, CNNTrainer.CONV_STRIDES_W, 1),
                   padding='SAME', name=None, ):
            # 如果使用默认步长strides[height, width]=[1,1], 则卷积后的图片大小不变
            return tf.nn.conv2d(x, W, strides, padding, name=name, )

        def max_pool(x, ksize,
                     strides=(1, CNNTrainer.POOL_STRIDES_H, CNNTrainer.POOL_STRIDES_W, 1),
                     padding='SAME', name=None, ):
            # 如果使用默认步长strides[height, width]=[2,2], 则卷积后的图片大小变为原来的一半
            return tf.nn.max_pool(x, ksize, strides, padding, name=name, )

        # First Convolutional Layer
        _height1, _width1 = initial_height, initial_width
        _in_channels1 = initial_channels
        x_image = tf.reshape(x, [-1, _height1, _width1, _in_channels1], name='x_image', )
        in1 = x_image  # shape[_example_cnt, _height1, _width1, _in_channels1]
        _out_channels1 = 32

        W_conv1 = weight_variable(
            [CNNTrainer.CONV_HEIGHT, CNNTrainer.CONV_WIDTH, _in_channels1, _out_channels1], name='W_conv1', )
        b_conv1 = bias_variable([_out_channels1], name='b_conv1', )

        h_conv1 = tf.nn.relu(conv2d(in1, W_conv1) + b_conv1, name='h_conv1', )
        h_pool1 = max_pool(h_conv1, CNNTrainer.POOL_SHAPE, name='h_pool1', )

        out1 = h_pool1

        # Second Convolutional Layer
        in2 = out1  # shape[_example_cnt, _height2, _width2, _in_channels2]
        _height2, _width2 = out1.get_shape()[1].value, out1.get_shape()[2].value
        _in_channels2 = out1.get_shape()[3].value
        _out_channels2 = 64

        W_conv2 = weight_variable(
            [CNNTrainer.CONV_HEIGHT, CNNTrainer.CONV_WIDTH, _in_channels2, _out_channels2], name='W_conv2', )
        b_conv2 = bias_variable([_out_channels2], name='b_conv2', )

        h_conv2 = tf.nn.relu(conv2d(in2, W_conv2) + b_conv2, name='h_conv2', )
        h_pool2 = max_pool(h_conv2, CNNTrainer.POOL_SHAPE, name='h_pool2', )

        out2 = h_pool2

        # Densely Connected Layer(Full Connected Layer)
        in3 = out2  # shape[_example_cnt, _height3, _width3, _in_channels3]
        _height3, _width3 = in3.get_shape()[1].value, in3.get_shape()[2].value
        _in_channels3 = in3.get_shape()[3].value
        _out_width3 = 1024

        W_fc1 = weight_variable([_height3 * _width3 * _in_channels3, _out_width3], name='W_fc1', )
        b_fc1 = bias_variable([_out_width3], name='b_fc1', )

        h_pool2_flat = tf.reshape(in3, [-1, _height3 * _width3 * _in_channels3], name='h_pool2_flat', )
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name='h_fc1', )  # 这不再卷积,直接矩阵乘

        out3 = h_fc1  # shape[_example_cnt, _out_width3]

        # Dropout
        tmp_in = out3
        h_fc1_drop = tf.nn.dropout(tmp_in, keep_prob, name='h_fc1_drop', )
        tmp_out = h_fc1_drop
        out3 = tmp_out

        # Readout Layer
        in4 = out3  # shape[_example_cnt, _in_width4]
        _in_width4 = in4.get_shape()[1].value
        _out_width4 = target_class_cnt

        W_fc2 = weight_variable([_in_width4, _out_width4], name='W_fc2', )
        b_fc2 = bias_variable([_out_width4], name='b_fc2', )

        y_conv = tf.add(tf.matmul(in4, W_fc2), b_fc2, name='y_conv', )

        out4 = y_conv  # shape[_example_cnt, _out_width4]

        # Train definition
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv), name='cross_entropy', )
        train_per_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, name='train_per_step', )

        # Evaluate definition
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1), name='correct_prediction', )
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy', )

        return train_per_step, accuracy

    @staticmethod
    def fit(argv):
        """支持训练并评估 baseline 级别的输入为任意<height, width, in_channels, target_class_cnt> 的 cnn 模型
        cnn的样本格式: 每行样本是一张拉成1维的图片(height*weight*in_channels), 外加 one_hot形式的标签(长度为 target_class_cnt )
            即,每行共有 height*weight*in_channels + target_class_cnt 列
        @:param argv, 长度为 12 的 list , 分别为<
                    delimiter,
                    train_data_x_file_path, train_data_y_file_path, test_data_x_file_path, test_data_y_file_path,
                    initial_height, initial_width, initial_channels, target_class_cnt,
                    iteration, batch_size,
                    model_file_path,
                >
        """
        # argv
        _offset, _length = 1, 1
        delimiter, = argv[_offset:_offset + _length]
        _offset, _length = _offset + _length, 4
        train_data_x_file_path, train_data_y_file_path, test_data_x_file_path, test_data_y_file_path = \
            argv[_offset:_offset + _length]
        _offset, _length = _offset + _length, 4
        initial_height, initial_width, initial_channels, target_class_cnt = map(int, argv[_offset:_offset + _length])
        _offset, _length = _offset + _length, 2
        iteration, batch_size = map(int, argv[_offset:_offset + _length])
        _offset, _length = _offset + _length, 1
        model_file_path, = argv[_offset:_offset + _length]

        # Construct
        # input and labels
        x = tf.placeholder(tf.float32, shape=[None, initial_height * initial_width], name='x', )
        y_ = tf.placeholder(tf.float32, shape=[None, target_class_cnt], name="y_", )
        keep_prob = tf.placeholder(tf.float32, name='keep_prob', )
        # trainer and evaluator
        train_per_step, accuracy = CNNTrainer.construct(
            initial_height, initial_width, initial_channels, target_class_cnt,
            x, y_, keep_prob,
        )

        with tf.Session() as sess:
            # load data
            train_data_x, train_data_y = map(lambda _: np.loadtxt(_, delimiter=delimiter),
                                             (train_data_x_file_path, train_data_y_file_path,))
            train_data = np.column_stack((train_data_x, train_data_y,))
            del train_data_x
            del train_data_y

            # train
            logging.info("start to train.")
            start_time = time.time()
            CNNTrainer.train(
                sess,
                train_per_step, accuracy,
                iteration, batch_size,
                train_data, target_class_cnt,
                x, y_, keep_prob,
            )
            del train_data
            end_time = time.time()
            logging.info("end to train.")
            logging.info('cost time: %.2fs' % (end_time - start_time))

            # dump model
            model_dir_path = CNNTrainer.dump(sess, model_file_path)
            logging.info("dump model into dir: %s" % model_dir_path)

        with tf.Session() as sess:
            # load data
            test_data_x, test_data_y = map(lambda _: np.loadtxt(_, delimiter=delimiter),
                                           (test_data_x_file_path, test_data_y_file_path,))
            test_data = np.column_stack((test_data_x, test_data_y,))
            del test_data_x
            del test_data_y

            # load model
            graph = CNNTrainer.load(sess, model_file_path)
            x = graph.get_tensor_by_name("x:0")
            y_ = graph.get_tensor_by_name("y_:0")
            keep_prob = graph.get_tensor_by_name("keep_prob:0")
            # evaluate
            train_accuracy = CNNTrainer.evaluate(
                sess,
                accuracy,
                test_data, target_class_cnt,
                x, y_, keep_prob,
            )
            del test_data
            logging.info("test accuracy %g" % train_accuracy)

    @staticmethod
    def evaluate(
            sess,
            accuracy,
            test_data, target_class_cnt,
            x, y_, keep_prob,
    ):
        _X, _Y = CNNTrainer.__format_inputs(test_data, target_class_cnt, )
        train_accuracy = accuracy.eval(feed_dict={x: _X, y_: _Y, keep_prob: 1.0}, session=sess)
        return train_accuracy

    @staticmethod
    def train(
            sess,
            train_per_step, accuracy=None,
            iteration=None, batch_size=None,
            train_data=None, target_class_cnt=None,
            x=None, y_=None, keep_prob=None,
    ):
        sess.run(tf.global_variables_initializer())
        for i in range(iteration):
            batch = random_sample(train_data, batch_size)
            # print progress
            if i % 100 == 0:
                train_accuracy = CNNTrainer.evaluate(
                    sess,
                    accuracy,
                    batch, target_class_cnt,
                    x, y_, keep_prob,
                )
                logging.info("step %d, training accuracy %g" % (i, train_accuracy))
            _X, _Y = CNNTrainer.__format_inputs(batch, target_class_cnt, )
            train_per_step.run(feed_dict={x: _X, y_: _Y, keep_prob: CNNTrainer.KEEP_PROB}, session=sess)
