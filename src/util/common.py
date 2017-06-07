# coding=utf-8
import random


def new_proportion(precision=100):
    """生成一个概率 p, 精度为 1/precision
    :param precision: int, 大于等于1
    :return:
    """
    precision = int(precision)
    assert precision >= 1
    return 1.0 * random.randint(0, precision - 1) / precision
