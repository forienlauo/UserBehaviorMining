# coding=utf-8
import logging
import os
import time

import numpy as np
import tensorflow as tf

import conf
from src.util.common import dump_model, load_model
from src.util.sampler import random_sample


class CNNTrainer(object):
    # cnn configuration
    CONV_STRIDES_H, CONV_STRIDES_W = 1, 1
    DEFAULT_CONV_HEIGHT, DEFAULT_CONV_WIDTH = 5, 5

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
    <conv_height> <conv_width>
    <neurons_nums>
    [cpu_core_num]
where
    neurons_nums is numbers of neurons in each conv layer, separated by comma(support no more than 3 conv layers)
""")

    @staticmethod
    def fit(argv):
        """训练并评估 cnn 模型
        样本格式(x): 每行样本是一张拉成1维的图片(height*weight*in_channels)
        标签格式(y): 每行标签是一个 one_hot 形式的向量(长度为 target_class_cnt )
        @:param argv list, 详见 print_usage() 方法
        """
        if len(argv) < 16:
            CNNTrainer.print_usage()
            return 1
        logging.info('argv: "%s"', ' '.join(argv))

        # argv
        # required
        _offset, _length = 0, 1
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
        _offset, _length = _offset + _length, 2
        conv_height, conv_width = map(int, argv[_offset:_offset + _length])
        _offset, _length = _offset + _length, 1
        _neurons_nums_str, = argv[_offset:_offset + _length]
        neurons_nums = map(int, str(_neurons_nums_str).strip().split(','))
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

        trainer, evaluator = CNNTrainer.construct(
            initial_height, initial_width, initial_channels, target_class_cnt,
            x, y_, keep_prob,
            conv_height, conv_width,
            neurons_nums,
        )

        # load data
        logging.info("start to load data.")
        start_time = time.time()
        _basedir_path = os.path.dirname(train_data_x_file_path)
        train_data = CNNTrainer.load_data(
            train_data_x_file_path, train_data_y_file_path,
            delimiter,
            os.path.join(_basedir_path, 'train_data.npy'),
        )
        test_data = CNNTrainer.load_data(
            test_data_x_file_path, test_data_y_file_path,
            delimiter,
            os.path.join(_basedir_path, 'test_data.npy'),
        )
        end_time = time.time()
        logging.info("end to load data.")
        logging.info('cost time: %.2fs' % (end_time - start_time))

        config = tf.ConfigProto(
            allow_soft_placement=True,
            # device_count={"CPU": cpu_core_num},
            # inter_op_parallelism_threads=cpu_core_num,
            # intra_op_parallelism_threads=cpu_core_num,
        )

        with tf.Session(config=config) as sess:
            # train
            logging.info("start to train.")
            start_time = time.time()
            trainer.train(
                sess, summary_log_dir_path,
                evaluator,
                iteration, batch_size,
                train_data, test_data, target_class_cnt,
                x, y_, keep_prob,
            )
            del train_data
            end_time = time.time()
            logging.info("end to train.")
            logging.info('cost time: %.2fs' % (end_time - start_time))

            # dump model
            dump_model(sess, model_file_path)
            logging.info("dump model into: %s" % model_file_path)

            # evaluate
            logging.info("start to evaluate.")
            start_time = time.time()
            test_data_len = len(test_data)
            evaluate_result = evaluator.evaluate(
                sess,
                batch_size,
                test_data, target_class_cnt,
                x, y_, keep_prob,
            )
            del test_data
            end_time = time.time()
            logging.info("end to evaluate.")
            logging.info('cost time: %.2fs' % (end_time - start_time,))
            logging.info('total test data: %d' % (test_data_len,))
            logging.info("final evaluate_result: %s" % (evaluate_result,))

        return 0

    @staticmethod
    def construct(
            initial_height, initial_width, initial_channels, target_class_cnt,
            x, y_, keep_prob,
            conv_height, conv_width,
            neurons_nums,
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
            return tf.nn.conv2d(x, W, strides, padding, name=name, )

        def max_pool(x, ksize,
                     strides=(1, CNNTrainer.POOL_STRIDES_H, CNNTrainer.POOL_STRIDES_W, 1),
                     padding='SAME', name=None, ):
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
                _in = out0
                _height, _width = _in.get_shape()[1].value, _in.get_shape()[2].value
                _in_channels = _in.get_shape()[3].value
                _out_channels = neurons_nums[0]

                W_conv = weight_variable(
                    [conv_height, conv_width, _in_channels, _out_channels], name='W_conv', )
                tf.summary.histogram('W_conv', W_conv)
                b_conv = bias_variable([_out_channels], name='b_conv', )
                tf.summary.histogram('b_conv', b_conv)
                h_conv = tf.nn.relu(conv2d(_in, W_conv) + b_conv, name='h_conv', )

                out = h_conv
                CNNTrainer.add_image2summary(out, 'out')

            # S2
            with tf.name_scope('S2') as _:
                _in = out

                h_pool = max_pool(_in, CNNTrainer.POOL_SHAPE, name='h_pool', )

                out = h_pool
                CNNTrainer.add_image2summary(out, 'out')

            # C3
            with tf.name_scope('C3') as _:
                _in = out
                _height, _width = _in.get_shape()[1].value, _in.get_shape()[2].value
                _in_channels = _in.get_shape()[3].value
                _out_channels = neurons_nums[1]

                W_conv = weight_variable(
                    [conv_height, conv_width, _in_channels, _out_channels], name='W_conv', )
                tf.summary.histogram('W_conv', W_conv)
                b_conv = bias_variable([_out_channels], name='b_conv', )
                tf.summary.histogram('b_conv', b_conv)
                h_conv = tf.nn.relu(conv2d(_in, W_conv) + b_conv, name='h_conv', )

                out = h_conv
                CNNTrainer.add_image2summary(out, 'out')

            # S4
            with tf.name_scope('S4') as _:
                _in = out

                h_pool = max_pool(_in, CNNTrainer.POOL_SHAPE, name='h_pool', )

                out = h_pool
                CNNTrainer.add_image2summary(out, 'out')

            # C5
            with tf.name_scope('C5') as _:
                _in = out
                _height, _width = _in.get_shape()[1].value, _in.get_shape()[2].value
                _in_channels = _in.get_shape()[3].value
                _out_channels = neurons_nums[2]

                W_conv = weight_variable(
                    [conv_height, conv_width, _in_channels, _out_channels], name='W_conv', )
                tf.summary.histogram('W_conv', W_conv)
                b_conv = bias_variable([_out_channels], name='b_conv', )
                tf.summary.histogram('b_conv', b_conv)
                h_conv = tf.nn.relu(conv2d(_in, W_conv) + b_conv, name='h_conv', )

                out = h_conv
                CNNTrainer.add_image2summary(out, 'out')

            # F6, Densely Connected Layer(Full Connected Layer)
            with tf.name_scope('F6') as _:
                _in = out
                _height, _width = _in.get_shape()[1].value, _in.get_shape()[2].value
                _in_channels = _in.get_shape()[3].value
                _out_width = 1024

                W_fc = weight_variable([_height * _width * _in_channels, _out_width], name='W_fc', )
                b_fc = bias_variable([_out_width], name='b_fc', )
                h_pool_flat = tf.reshape(_in, [-1, _height * _width * _in_channels], name='h_pool_flat', )
                h_fc = tf.nn.relu(tf.matmul(h_pool_flat, W_fc) + b_fc, name='h_fc', )

                out = h_fc

            # Dropout Layer
            with tf.name_scope('DropoutLayer') as _:
                _in = out

                h_fc_drop = tf.nn.dropout(_in, keep_prob, name='h_fc_drop', )

                out = h_fc_drop

            # Output Layer
            with tf.name_scope('OutputLayer') as _:
                _in_ = out
                _in_width_ = _in_.get_shape()[1].value
                _out_width_ = target_class_cnt

                W_fc_ = weight_variable([_in_width_, _out_width_], name='W_fc_', )
                b_fc_ = bias_variable([_out_width_], name='b_fc_', )
                y = tf.add(tf.matmul(_in_, W_fc_), b_fc_, name='y', )

        # Trainer
        with tf.name_scope('trainer') as _:
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y), name='loss', )
            train_per_step = tf.train.AdamOptimizer(1e-5).minimize(loss, name='train_per_step', )
            tf.summary.scalar('loss', loss)

            trainer = CNNTrainer.Trainer(train_per_step)

        # Evaluator
        with tf.name_scope('evaluator') as _:
            example_cnt = tf.count_nonzero(
                tf.logical_or(tf.cast(tf.argmax(y_, 1), dtype=tf.bool), True), name='example_cnt')  # 样本总数

            _correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            correct_cnt = tf.count_nonzero(_correct_prediction, name='correct_cnt')  # 将正样本预测为正,负样本预测为负的数量
            accuracy = tf.reduce_mean(tf.cast(_correct_prediction, tf.float32), name='accuracy', )
            tf.summary.scalar('accuracy', accuracy)

            # 该模型初步处理2种用户类型，正常、非正常用户，正常用户序号为0，异常用户序号为1
            # 关心异常用户的查全率、查准率
            TARGET_LABEL_IDX = 1  # 目标类别(假设为正)的索引
            _right_label = TARGET_LABEL_IDX
            _true_right = tf.equal(tf.argmax(y_, 1), _right_label)
            _predicted_right = tf.equal(tf.argmax(y, 1), _right_label)
            both_right_cnt = tf.count_nonzero(
                tf.logical_and(_true_right, _predicted_right), name='both_right_cnt')  # 将正样本预测为正的数量
            true_right_cnt = tf.count_nonzero(
                _true_right, name='true_right_cnt')  # 正样本的数量
            predicted_right_cnt = tf.count_nonzero(_predicted_right, name='predicted_right_cnt')  # 预测为正样本的数量
            recall = -1.0 if true_right_cnt == 0 else \
                tf.divide(tf.to_float(both_right_cnt), tf.to_float(true_right_cnt), name='recall')
            precision = -1.0 if predicted_right_cnt == 0 else \
                tf.divide(tf.to_float(both_right_cnt), tf.to_float(predicted_right_cnt), name='precision')
            tf.summary.scalar('recall', recall)
            tf.summary.scalar('precision', precision)

            evaluator = CNNTrainer.Evaluator(
                accuracy, recall, precision,
                example_cnt, correct_cnt, both_right_cnt,
                true_right_cnt, predicted_right_cnt,
            )

        return trainer, evaluator

    @staticmethod
    def format_inputs(example, target_class_cnt):
        example = np.array(example)
        return example[:, :-target_class_cnt], example[:, -target_class_cnt:]

    @staticmethod
    def load_data(
            data_x_file_path, data_y_file_path,
            delimiter,
            cache=None,
    ):
        if cache is None or not os.path.exists(cache):
            # TODO 减小数据加载的内存占用(如预先shuffle,然后顺序yield读)
            data_x, data_y = map(lambda _: np.loadtxt(_, delimiter=delimiter),
                                 (data_x_file_path, data_y_file_path,))
            data = np.column_stack((data_x, data_y,))
            del data_x
            del data_y
            if cache is not None:
                np.save(cache, data)
        else:
            data = np.load(cache)
        return data

    @staticmethod
    def view_evaluate_result(
            delimiter,
            test_data_x_file_path, test_data_y_file_path,
            target_class_cnt,
            batch_size,
            model_file_path,
            cpu_core_num=conf.CPU_COUNT,
    ):
        # load data
        logging.info("start to load data.")
        start_time = time.time()
        _basedir_path = os.path.dirname(test_data_x_file_path)
        test_data = CNNTrainer.load_data(
            test_data_x_file_path, test_data_y_file_path,
            delimiter,
            os.path.join(_basedir_path, 'test_data.npy'),
        )
        end_time = time.time()
        logging.info("end to load data.")
        logging.info('cost time: %.2fs' % (end_time - start_time))

        config = tf.ConfigProto(
            allow_soft_placement=True,
            # device_count={"CPU": cpu_core_num},
            # inter_op_parallelism_threads=cpu_core_num,
            # intra_op_parallelism_threads=cpu_core_num,
        )

        with tf.Session(config=config) as sess:
            # load model
            graph = load_model(sess, model_file_path)
            x = graph.get_tensor_by_name("input/x:0")
            y_ = graph.get_tensor_by_name("input/y_:0")
            keep_prob = graph.get_tensor_by_name("input/keep_prob:0")

            accuracy = graph.get_tensor_by_name("evaluator/accuracy:0")
            recall = graph.get_tensor_by_name("evaluator/recall:0")
            precision = graph.get_tensor_by_name("evaluator/precision:0")

            example_cnt = graph.get_tensor_by_name("evaluator/example_cnt:0")
            correct_cnt = graph.get_tensor_by_name("evaluator/correct_cnt:0")
            both_right_cnt = graph.get_tensor_by_name("evaluator/both_right_cnt:0")
            true_right_cnt = graph.get_tensor_by_name("evaluator/true_right_cnt:0")
            predicted_right_cnt = graph.get_tensor_by_name("evaluator/predicted_right_cnt:0")

            logging.info("load model from: %s" % model_file_path)

            # evaluate
            logging.info("start to evaluate.")
            start_time = time.time()
            test_data_len = len(test_data)
            evaluator = CNNTrainer.Evaluator(
                accuracy, recall, precision,
                example_cnt, correct_cnt, both_right_cnt,
                true_right_cnt, predicted_right_cnt,
            )
            evaluate_result = evaluator.evaluate(
                sess,
                batch_size,
                test_data, target_class_cnt,
                x, y_, keep_prob
            )
            del test_data
            end_time = time.time()
            logging.info("end to evaluate.")
            logging.info('cost time: %.2fs' % (end_time - start_time,))
            logging.info('total data: %d' % (test_data_len,))
            logging.info("evaluate result %s" % (evaluate_result,))

    @staticmethod
    def add_image2summary(x, image_name_prefix):
        channels = x.get_shape()[3].value
        for channel_no in range(channels):
            image = x[:, :, :, channel_no:channel_no + 1]
            image_name = '%s-%d' % (image_name_prefix, channel_no,)
            tf.summary.image(image_name, image)

    class Trainer(object):
        PRINT_PROGRESS_PER_STEP_NUM = 100

        def __init__(self, train_per_step, ):
            super(CNNTrainer.Trainer, self).__init__()
            self.train_per_step = train_per_step

        def train(
                self,
                sess, summary_log_dir_path,
                evaluator=None,
                iteration=None, batch_size=None,
                train_data=None, test_data=None, target_class_cnt=None,
                x=None, y_=None, keep_prob=None,
        ):
            train_per_step = self.train_per_step

            summaries = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(logdir=summary_log_dir_path, graph=sess.graph)
            sess.run(tf.global_variables_initializer())
            for i in range(iteration):
                batch_train = random_sample(train_data, batch_size)
                _X_train, _Y_train = CNNTrainer.format_inputs(batch_train, target_class_cnt, )
                # print progress
                if evaluator is not None:
                    if i % CNNTrainer.Trainer.PRINT_PROGRESS_PER_STEP_NUM == 0:
                        train_evl_rs = evaluator.evaluate_one(
                            sess,
                            batch_train, target_class_cnt,
                            x, y_, keep_prob
                        )
                        batch_test = random_sample(test_data, 2 * batch_size)
                        test_evl_rs = evaluator.evaluate_one(
                            sess,
                            batch_test, target_class_cnt,
                            x, y_, keep_prob
                        )
                        logging.info(
                            "step %d, training evaluate_result: %s, testing evaluate_result: %s"
                            % (i, train_evl_rs, test_evl_rs))
                        # TODO 注释掉下段代码。实验阶段应该关注各评估指标的变化,上线时可根据实际效果选择是否打开
                        # accuracy_threshold = 0.83
                        # if test_evl_rs.accuracy_ratio > accuracy_threshold \
                        #         and train_evl_rs.accuracy_ratio > accuracy_threshold:
                        #     logging.info(
                        #         "exiting for reason: both train_accuracy and test_accuracy gt accuracy_threshold(%s)"
                        #         % (accuracy_threshold,))
                        #     return
                feed = {x: _X_train, y_: _Y_train, keep_prob: CNNTrainer.KEEP_PROB}
                train_per_step.run(feed_dict=feed, session=sess)
                summaries_result = sess.run(summaries, feed_dict=feed, )
                summary_writer.add_summary(summaries_result, global_step=i)

            summary_writer.close()

    class Evaluator(object):

        def __init__(
                self,
                accuracy, recall, precision,
                example_cnt, correct_cnt, both_right_cnt,
                true_right_cnt, predicted_right_cnt,
        ):
            super(CNNTrainer.Evaluator, self).__init__()
            self.accuracy = accuracy
            self.recall = recall
            self.precision = precision
            self.example_cnt = example_cnt
            self.correct_cnt = correct_cnt
            self.both_right_cnt = both_right_cnt
            self.true_right_cnt = true_right_cnt
            self.predicted_right_cnt = predicted_right_cnt

        def evaluate(
                self,
                sess,
                batch_size,
                data, target_class_cnt,
                x, y_, keep_prob,
        ):
            iteration = int(len(data) / batch_size) + 1
            sum_example_cnt = 0
            sum_both_right_cnt = 0
            sum_correct_cnt = 0
            sum_true_right_cnt = 0
            sum_predicted_right_cnt = 0
            for i in range(iteration):
                batch_test = random_sample(data, batch_size)
                _result = self.evaluate_one(
                    sess,
                    batch_test, target_class_cnt,
                    x, y_, keep_prob
                )
                sum_example_cnt += _result.example_cnt
                sum_correct_cnt += _result.correct_cnt
                sum_both_right_cnt += _result.both_right_cnt
                sum_true_right_cnt += _result.true_right_cnt
                sum_predicted_right_cnt += _result.predicted_right_cnt
            final_accuracy = -1.0 if sum_example_cnt == 0 else \
                1.0 * sum_correct_cnt / sum_example_cnt
            final_recall = -1.0 if sum_true_right_cnt == 0 else \
                1.0 * sum_both_right_cnt / sum_true_right_cnt
            final_precision = -1.0 if sum_predicted_right_cnt == 0 else \
                1.0 * sum_both_right_cnt / sum_predicted_right_cnt
            result = CNNTrainer.Evaluator.Result(
                final_accuracy, final_recall, final_precision,
                sum_example_cnt, sum_correct_cnt, sum_both_right_cnt,
                sum_true_right_cnt, sum_predicted_right_cnt,
            )
            return result

        def evaluate_one(
                self,
                sess,
                data, target_class_cnt,
                x, y_, keep_prob,
        ):
            _X, _Y = CNNTrainer.format_inputs(data, target_class_cnt, )
            feed_dict = {x: _X, y_: _Y, keep_prob: 1.0}
            accuracy_ratio = self.accuracy.eval(feed_dict=feed_dict, session=sess)
            recall_ratio = self.recall.eval(feed_dict=feed_dict, session=sess)
            precision_ratio = self.precision.eval(feed_dict=feed_dict, session=sess)
            example_cnt = self.example_cnt.eval(feed_dict=feed_dict, session=sess)
            correct_cnt = self.correct_cnt.eval(feed_dict=feed_dict, session=sess)
            both_right_cnt = self.both_right_cnt.eval(feed_dict=feed_dict, session=sess)
            true_right_cnt = self.true_right_cnt.eval(feed_dict=feed_dict, session=sess)
            predicted_right_cnt = self.predicted_right_cnt.eval(feed_dict=feed_dict, session=sess)
            result = CNNTrainer.Evaluator.Result(
                accuracy_ratio, recall_ratio, precision_ratio,
                example_cnt, correct_cnt, both_right_cnt,
                true_right_cnt, predicted_right_cnt,
            )
            return result

        class Result(object):
            def __init__(
                    self,
                    accuracy_ratio, recall_ratio, precision_ratio,
                    example_cnt, correct_cnt, both_right_cnt,
                    true_right_cnt, predicted_right_cnt,
            ):
                super(CNNTrainer.Evaluator.Result, self).__init__()
                self.accuracy_ratio = accuracy_ratio
                self.recall_ratio = recall_ratio
                self.precision_ratio = precision_ratio
                self.example_cnt = example_cnt
                self.correct_cnt = correct_cnt
                self.both_right_cnt = both_right_cnt
                self.true_right_cnt = true_right_cnt
                self.predicted_right_cnt = predicted_right_cnt

            def __str__(self):
                return "result {accuracy: %g, recall: %g, precision: %g, " \
                       "example_cnt: %g, correct_cnt: %g, both_right_cnt: %g, " \
                       "true_right_cnt: %g, predicted_right_cnt: %g}" \
                       % (self.accuracy_ratio, self.recall_ratio, self.precision_ratio,
                          self.example_cnt, self.correct_cnt, self.both_right_cnt,
                          self.true_right_cnt, self.predicted_right_cnt,)
