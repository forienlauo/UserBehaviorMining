# coding=utf-8
import logging
import os
import time

import numpy as np
import tensorflow as tf

import conf
from src.util.sampler import random_sample


class CNNTrainer(object):
    # cnn configuration
    CONV_STRIDES_H, CONV_STRIDES_W = 1, 1
    CONV_HEIGHT, CONV_WIDTH = 5, 5

    POOL_STRIDES_H, POOL_STRIDES_W = 2, 2
    POOL_SHAPE = [1, 2, 2, 1]

    KEEP_PROB = 0.4

    @staticmethod
    def print_usage():
        print("""
need args:
    <delimiter>
    <train_data_x_file_path> <train_data_y_file_path> <test_data_x_file_path> <test_data_y_file_path>
    <initial_height> <initial_width> <initial_channels> <target_class_cnt>
    <iteration> <batch_size>
    <model_file_path>
    <summary_log_dir_path>
    [cpu_core_num]
""")

    @staticmethod
    def fit(argv):
        """支持训练并评估 baseline 级别的输入为任意<height, width, in_channels, target_class_cnt> 的 cnn 模型
        cnn的样本格式: 每行样本是一张拉成1维的图片(height*weight*in_channels), 外加 one_hot形式的标签(长度为 target_class_cnt )
            即,每行共有 height*weight*in_channels + target_class_cnt 列
        @:param argv list,
                    argv[0]: 启动文件名;
                    argv[1:14]为必选项, <
                        delimiter,
                        train_data_x_file_path, train_data_y_file_path, test_data_x_file_path, test_data_y_file_path,
                        initial_height, initial_width, initial_channels, target_class_cnt,
                        iteration, batch_size,
                        model_file_path,
                        summary_log_dir_path,
                    >;
                    argv[14:]为可选项, [
                        cpu_core_num,
                    ]
        """
        if len(argv[1:]) < 13:
            CNNTrainer.print_usage()
            return 1

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
        _offset, _length = _offset + _length, 1
        summary_log_dir_path, = argv[_offset:_offset + _length]
        # optional
        _offset, _length = _offset + _length, 1
        cpu_core_num = conf.CPU_COUNT
        if len(argv) > _offset:
            cpu_core_num, = map(int, argv[_offset:_offset + _length])

        # Construct
        # input and labels
        with tf.name_scope('input') as _:
            x = tf.placeholder(tf.float32, shape=[None, initial_height * initial_width], name='x', )
            y_ = tf.placeholder(tf.float32, shape=[None, target_class_cnt], name="y_", )
            keep_prob = tf.placeholder(tf.float32, name='keep_prob', )
        # trainer and evaluator
        train_per_step, accuracy = CNNTrainer.construct(
            initial_height, initial_width, initial_channels, target_class_cnt,
            x, y_, keep_prob,
        )

        # load data
        train_data, test_data = CNNTrainer.load_data(
            train_data_x_file_path, train_data_y_file_path,
            test_data_x_file_path, test_data_y_file_path,
            delimiter,
        )

        config = tf.ConfigProto(
            device_count={"CPU": cpu_core_num},
            inter_op_parallelism_threads=cpu_core_num,
            intra_op_parallelism_threads=cpu_core_num,
        )

        with tf.Session(config=config) as sess:
            # train
            logging.info("start to train.")
            start_time = time.time()
            CNNTrainer.train(
                sess, summary_log_dir_path,
                train_per_step, accuracy,
                iteration, batch_size,
                train_data, test_data, target_class_cnt,
                x, y_, keep_prob,
            )
            del train_data
            end_time = time.time()
            logging.info("end to train.")
            logging.info('cost time: %.2fs' % (end_time - start_time))

            # dump model
            CNNTrainer.dump_model(sess, model_file_path)
            logging.info("dump model into: %s" % model_file_path)

        with tf.Session(config=config) as sess:
            # load model
            graph = CNNTrainer.load_model(sess, model_file_path)
            x = graph.get_tensor_by_name("x:0")
            y_ = graph.get_tensor_by_name("y_:0")
            keep_prob = graph.get_tensor_by_name("keep_prob:0")
            logging.info("load model from: %s" % model_file_path)

            # evaluate
            logging.info("start to train.")
            start_time = time.time()
            iteration_test = int(len(test_data) / batch_size) + 1
            tmp_sum_accuracy = 0
            for i in range(iteration_test):
                batch_test = random_sample(test_data, batch_size)
                tmp_sum_accuracy += CNNTrainer.evaluate(
                    sess,
                    accuracy,
                    batch_test, target_class_cnt,
                    x, y_, keep_prob,
                )
            test_data_len = len(test_data)
            del test_data
            final_accuracy = tmp_sum_accuracy / iteration_test
            end_time = time.time()
            logging.info("end to evaluate.")
            logging.info('cost time: %.2fs' % (end_time - start_time,))
            logging.info('total data: %d' % (test_data_len,))
            logging.info("final test accuracy %g" % (final_accuracy,))

        return 0

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

        with tf.name_scope('model') as _:
            # Input Layer
            with tf.name_scope('InputLayer') as _:
                in0 = x
                x_image = tf.reshape(in0, [-1, initial_height, initial_width, initial_channels], name='x_image', )
                out0 = x_image
                CNNTrainer.add_image2summary(out0, 'out0')

            # C1
            with tf.name_scope('C1') as _:
                in1 = out0  # shape[_example_cnt, _height1, _width1, _in_channels1]
                _height1, _width1 = in1.get_shape()[1].value, in1.get_shape()[2].value
                _in_channels1 = in1.get_shape()[3].value
                _out_channels1 = 32

                W_conv1 = weight_variable(
                    [CNNTrainer.CONV_HEIGHT, CNNTrainer.CONV_WIDTH, _in_channels1, _out_channels1], name='W_conv1', )
                tf.summary.histogram('W_conv1', W_conv1)
                b_conv1 = bias_variable([_out_channels1], name='b_conv1', )
                tf.summary.histogram('b_conv1', b_conv1)
                h_conv1 = tf.nn.relu(conv2d(in1, W_conv1) + b_conv1, name='h_conv1', )

                out1 = h_conv1
                CNNTrainer.add_image2summary(out1, 'out1')

            # S2
            with tf.name_scope('S2') as _:
                in2 = out1

                h_pool2 = max_pool(in2, CNNTrainer.POOL_SHAPE, name='h_pool2', )

                out2 = h_pool2
                CNNTrainer.add_image2summary(out2, 'out2')

            # C3
            with tf.name_scope('C3') as _:
                in3 = out2  # shape[_example_cnt, _height2, _width2, _in_channels2]
                _height2, _width2 = in3.get_shape()[1].value, in3.get_shape()[2].value
                _in_channels2 = in3.get_shape()[3].value
                _out_channels2 = 64

                W_conv2 = weight_variable(
                    [CNNTrainer.CONV_HEIGHT, CNNTrainer.CONV_WIDTH, _in_channels2, _out_channels2], name='W_conv2', )
                tf.summary.histogram('W_conv2', W_conv2)
                b_conv2 = bias_variable([_out_channels2], name='b_conv2', )
                tf.summary.histogram('b_conv2', b_conv2)
                h_conv2 = tf.nn.relu(conv2d(in3, W_conv2) + b_conv2, name='h_conv2', )

                out3 = h_conv2
                CNNTrainer.add_image2summary(out3, 'out3')

            # S4
            with tf.name_scope('S4') as _:
                in4 = out3

                h_pool2 = max_pool(in4, CNNTrainer.POOL_SHAPE, name='h_pool2', )

                out4 = h_pool2
                CNNTrainer.add_image2summary(out4, 'out4')

            # F5, Densely Connected Layer(Full Connected Layer)
            with tf.name_scope('F5') as _:
                in5 = out4  # shape[_example_cnt, _height3, _width3, _in_channels3]
                _height3, _width3 = in5.get_shape()[1].value, in5.get_shape()[2].value
                _in_channels3 = in5.get_shape()[3].value
                _out_width3 = 1024

                W_fc1 = weight_variable([_height3 * _width3 * _in_channels3, _out_width3], name='W_fc1', )
                b_fc1 = bias_variable([_out_width3], name='b_fc1', )
                h_pool2_flat = tf.reshape(in5, [-1, _height3 * _width3 * _in_channels3], name='h_pool2_flat', )
                h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name='h_fc1', )  # 这不再卷积,直接矩阵乘

                out5 = h_fc1  # shape[_example_cnt, _out_width3]

            # D6, Dropout
            with tf.name_scope('D6') as _:
                in6 = out5

                h_fc1_drop = tf.nn.dropout(in6, keep_prob, name='h_fc1_drop', )

                out6 = h_fc1_drop

            # Output Layer
            with tf.name_scope('OutputLayer') as _:
                in7 = out6  # shape[_example_cnt, _in_width4]
                _in_width4 = in7.get_shape()[1].value
                _out_width4 = target_class_cnt
                W_fc2 = weight_variable([_in_width4, _out_width4], name='W_fc2', )
                b_fc2 = bias_variable([_out_width4], name='b_fc2', )
                y = tf.add(tf.matmul(in7, W_fc2), b_fc2, name='y', )

            out7 = y  # shape[_example_cnt, _out_width4]

        with tf.name_scope('trainer') as _:
            # Train definition
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y), name='loss', )
            train_per_step = tf.train.AdamOptimizer(1e-5).minimize(loss, name='train_per_step', )
            loss_summary = tf.summary.scalar('loss', loss)

        with tf.name_scope('evaluator') as _:
            # Evaluate definition
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1), name='correct_prediction', )
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy', )
            accuracy_summary = tf.summary.scalar('accuracy', accuracy)

        return train_per_step, accuracy

    @staticmethod
    def train(
            sess, summary_log_dir_path,
            train_per_step, accuracy=None,
            iteration=None, batch_size=None,
            train_data=None, test_data=None, target_class_cnt=None,
            x=None, y_=None, keep_prob=None,
    ):
        summaries = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir=summary_log_dir_path, graph=sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range(iteration):
            batch_train = random_sample(train_data, batch_size)
            batch_test = random_sample(test_data, 2 * batch_size)
            # print progress
            if accuracy is not None:
                if i % 100 == 0:
                    train_accuracy = CNNTrainer.evaluate(
                        sess,
                        accuracy,
                        batch_train, target_class_cnt,
                        x, y_, keep_prob,
                    )
                    test_accuracy = CNNTrainer.evaluate(
                        sess,
                        accuracy,
                        batch_test, target_class_cnt,
                        x, y_, keep_prob,
                    )
                    logging.info(
                        "step %d, training accuracy %g, testing accuracy %g" % (i, train_accuracy, test_accuracy))
                    if test_accuracy > 0.83 and train_accuracy > 0.83:  return
            _X, _Y = CNNTrainer.__format_inputs(batch_train, target_class_cnt, )
            train_per_step.run(feed_dict={x: _X, y_: _Y, keep_prob: CNNTrainer.KEEP_PROB}, session=sess)
            summaries_result = sess.run(summaries, feed_dict={x: _X, y_: _Y, keep_prob: CNNTrainer.KEEP_PROB}, )
            summary_writer.add_summary(summaries_result, global_step=i)
        summary_writer.close()

    @staticmethod
    def evaluate(
            sess,
            accuracy,
            data, target_class_cnt,
            x, y_, keep_prob,
    ):
        _X, _Y = CNNTrainer.__format_inputs(data, target_class_cnt, )
        test_accuracy = accuracy.eval(feed_dict={x: _X, y_: _Y, keep_prob: 1.0}, session=sess)
        return test_accuracy

    @staticmethod
    def __format_inputs(example, target_class_cnt):
        example = np.array(example)
        return example[:, :-target_class_cnt], example[:, -target_class_cnt:]

    @staticmethod
    def load_data(
            train_data_x_file_path, train_data_y_file_path,
            test_data_x_file_path, test_data_y_file_path,
            delimiter,
    ):
        basedir_path = os.path.dirname(train_data_x_file_path)
        train_data_cache = os.path.join(basedir_path, 'train_data.npy')
        test_data_cache = os.path.join(basedir_path, 'test_data.npy')
        logging.info("start to load data.")
        start_time = time.time()
        if os.path.exists(train_data_cache):
            train_data = np.load(train_data_cache)
            test_data = np.load(test_data_cache)
        else:
            train_data_x, train_data_y = map(lambda _: np.loadtxt(_, delimiter=delimiter),
                                             (train_data_x_file_path, train_data_y_file_path,))
            train_data = np.column_stack((train_data_x, train_data_y,))
            del train_data_x
            del train_data_y
            np.save(train_data_cache, train_data)

            test_data_x, test_data_y = map(lambda _: np.loadtxt(_, delimiter=delimiter),
                                           (test_data_x_file_path, test_data_y_file_path,))
            test_data = np.column_stack((test_data_x, test_data_y,))
            del test_data_x
            del test_data_y
            np.save(test_data_cache, test_data)
        end_time = time.time()
        logging.info("end to load data.")
        logging.info('cost time: %.2fs' % (end_time - start_time))
        return train_data, test_data

    @staticmethod
    def dump_model(sess, model_file_path, ):
        """保存一个sess中的全部变量
        实际并不存在文件路径 model_file_path, dirname(model_file_path) 作为保存的目录, basename(model_file_path) 作为模型的名字
        @:return 模型保存保存的目录,即 dirname(model_file_path)
        """
        saver = tf.train.Saver()
        saver.save(sess, model_file_path)

        model_dir = os.path.dirname(model_file_path)
        return model_dir

    @staticmethod
    def load_model(sess, model_file_path, ):
        """加载一个sess中的全部变量
        实际并不存在文件路径 model_file_path, dirname(model_file_path) 作为保存的目录, basename(model_file_path) 作为模型的名字
        @:return sess中的 gragh, 可以通过 graph 取得所有 tensor 和 operation
        """
        meta_gragh_path = '%s.meta' % (model_file_path,)
        saver = tf.train.import_meta_graph(meta_gragh_path)
        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(model_file_path)))

        return tf.get_default_graph()

    @staticmethod
    def add_image2summary(x, image_name_prefix):
        channels = x.get_shape()[3].value
        for channel_no in range(channels):
            image = x[:, :, :, channel_no:channel_no + 1]
            image_name = '%s-%d' % (image_name_prefix, channel_no,)
            tf.summary.image(image_name, image)
