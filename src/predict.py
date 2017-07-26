# coding=utf-8
import logging
import os
import time

import numpy as np
import tensorflow as tf

import conf
from src.util.common import load_model


class CNNPredictor(object):
    @staticmethod
    def print_usage():
        print("""
need args:
    <delimiter>
    <data_x_dir_path> <data_y_dir_path>
    <model_file_path>
    [cpu_core_num]
""")

    @staticmethod
    def fit(argv):
        """批量预测指定用户的标签
        data_x_dir_path: 存储待预测样本的目录. 目录中每个文件中存有一个用户对应的所有图片, 文件名是 <uid>.*, 文件中每行一个样本(x)
        样本格式(x): 每行样本是一张拉成1维的图片(<height, weight, in_channels>同训练数据)
        标签格式(y): 每行标签是一个整数, 表示标签号(不再是 one_hot 形式的向量(<target_class_cnt>同训练数据))
        @:param argv list, 详见 print_usage() 方法
        """
        if len(argv) < 4:
            CNNPredictor.print_usage()
            return 1
        logging.info('argv: "%s"', ' '.join(argv))

        # argv
        # required
        _offset, _length = 0, 1
        delimiter, = argv[_offset:_offset + _length]
        _offset, _length = _offset + _length, 2
        data_x_dir_path, data_y_dir_path = \
            argv[_offset:_offset + _length]
        _offset, _length = _offset + _length, 1
        model_file_path, = argv[_offset:_offset + _length]
        # optional
        _offset, _length = _offset + _length, 1
        cpu_core_num = conf.CPU_COUNT
        if len(argv) > _offset:
            cpu_core_num, = map(int, argv[_offset:_offset + _length])

        # Construct
        predictor = CNNPredictor.Predictor(model_file_path, cpu_core_num, )

        if not os.path.exists(data_y_dir_path):
            os.mkdir(data_x_dir_path)

        # batching predict
        predictor.init()
        for data_x_file_name in os.listdir(data_x_dir_path):
            uid = data_x_file_name.split(".")[0]
            data_x_file_path = os.path.join(data_x_dir_path, data_x_file_name)

            # load data
            logging.info("start to load data.")
            start_time = time.time()
            _basedir_path = os.path.dirname(data_x_file_path)
            data_x = CNNPredictor.load_data(
                data_x_file_path,
                delimiter,
                os.path.join(_basedir_path, 'data_x.npy'),
            )
            end_time = time.time()
            logging.info("end to load data.")
            logging.info('cost time: %.2fs' % (end_time - start_time))

            # predict
            logging.info("start to predict.")
            start_time = time.time()
            label = predictor.predict(data_x)
            end_time = time.time()
            logging.info("end to predict.")
            logging.info('cost time: %.2fs' % (end_time - start_time,))
            logging.info('predict label: %s' % (label,))

            del data_x

            data_y_file_path = os.path.join(data_y_dir_path, "%s.label" % str(uid))
            with open(data_y_file_path, "w") as data_y_file:
                data_y_file.write(str(label))

        predictor.close()

        return 0

    @staticmethod
    def load_data(
            data_x_file_path,
            delimiter,
            cache=None,
    ):
        if cache is None or not os.path.exists(cache):
            data_x = np.loadtxt(data_x_file_path, delimiter=delimiter)
            if cache is not None:
                np.save(cache, data_x)
        else:
            data_x = np.load(cache)
        return data_x

    class Predictor(object):

        def __init__(self, model_file_path, cpu_core_num=conf.CPU_COUNT, ):
            super(CNNPredictor.Predictor, self).__init__()
            self.__model_file_path = model_file_path
            self.__cpu_core_num = cpu_core_num

        def init(self):
            # open session
            config = tf.ConfigProto(
                device_count={"CPU": self.__cpu_core_num},
                inter_op_parallelism_threads=self.__cpu_core_num,
                intra_op_parallelism_threads=self.__cpu_core_num,
            )
            self.__sess = tf.Session(config=config)
            # load model
            self.__graph = load_model(self.__sess, self.__model_file_path)
            logging.info("load model from: %s" % self.__model_file_path)

        def close(self):
            # close session
            if self.__sess is not None:
                try:
                    self.__sess.close()
                except Exception, ignore:
                    pass

        def predict(self, data_x, ):
            x = self.__graph.get_tensor_by_name("input/x:0")
            keep_prob = self.__graph.get_tensor_by_name("input/keep_prob:0")
            y = self.__graph.get_tensor_by_name("model/OutputLayer/y:0")

            # assume that input/x belong to one users, so he or she should get one predicting-label
            label = tf.argmax(tf.reduce_sum(tf.nn.softmax(y), axis=0), axis=0)
            pre_label = label.eval(feed_dict={x: data_x, keep_prob: 1.0}, session=self.__sess)

            return int(pre_label)
