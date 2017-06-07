# coding=utf-8
import random

import numpy as np


def random_sample(population, k):
    """从 population 中随机抽样 k 个元素
    :param population: list, tuple, or numpy.ndarray
    :param k: int
    :return: return numpy.ndarray if population isinstance of numpy.ndarray, else list
    """
    sample_result = random.sample(population, k)
    if isinstance(population, np.ndarray):
        sample_result = np.array(sample_result)
    return sample_result
