# coding=utf-8
import random
import os
import tensorflow as tf


def new_proportion(precision=100):
    """生成一个概率 p, 精度为 1/precision
    :param precision: int, 大于等于1
    :return:
    """
    precision = int(precision)
    assert precision >= 1
    return 1.0 * random.randint(0, precision - 1) / precision


def dump_model(sess, model_file_path, ):
    """保存一个sess中的全部变量
    实际并不存在文件路径 __model_file_path, dirname(__model_file_path) 作为保存的目录, basename(__model_file_path) 作为模型的名字
    @:return 模型保存保存的目录,即 dirname(__model_file_path)
    """
    saver = tf.train.Saver()
    saver.save(sess, model_file_path)

    model_dir = os.path.dirname(model_file_path)
    return model_dir


def load_model(sess, model_file_path, ):
    """加载一个sess中的全部变量
    实际并不存在文件路径 __model_file_path, dirname(__model_file_path) 作为保存的目录, basename(__model_file_path) 作为模型的名字
    @:return sess中的 gragh, 可以通过 __graph 取得所有 tensor 和 operation
    """
    meta_gragh_path = '%s.meta' % (model_file_path,)
    saver = tf.train.import_meta_graph(meta_gragh_path)
    saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(model_file_path)))

    return tf.get_default_graph()
