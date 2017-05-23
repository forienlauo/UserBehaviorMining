import random

import numpy as np


def random_sample(population, k):
    sample_result = random.sample(population, k)
    if isinstance(population, np.ndarray):
        sample_result = np.array(sample_result)
    return sample_result
