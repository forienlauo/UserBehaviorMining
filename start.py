# coding=utf-8
import sys

import conf
from src.train import CNNTrainer
from src.util.example_utils import ExampleAllocator

if __name__ == '__main__':
    generatedDataFilePaths = ('resource/little_data/data/data_x.txt', 'resource/little_data/data/data_y.txt',)
    trainDataFilePaths = ('resource/little_data/data/train_data_x.txt', 'resource/little_data/data/train_data_y.txt',)
    testDataFilePaths = ('resource/little_data/data/test_data_x.txt', 'resource/little_data/data/test_data_y.txt',)
    sys.argv.extend([
        # cnn argv
        # # test
        # '\t',
        # 'tmp/cnn_test/train_data_x.in', 'tmp/cnn_test/train_data_y.in',
        # 'tmp/cnn_test/test_data_x.in', 'tmp/cnn_test/test_data_y.in',
        # 5 * 60, 20, 1, 2,
        # 200, 10,
        # 'tmp/model_persistence/cnn_test.model',
        # little data
        ',',
        trainDataFilePaths[0], trainDataFilePaths[1],
        testDataFilePaths[0], testDataFilePaths[1],
        60, 36, 1, 2,
        1000, 20,
        'resource/little_data/model/cnn.model',
        2,
    ])
    argv = map(lambda x: str(x), sys.argv)

    conf.init()

    # ExampleAllocator(
    #     generatedDataFilePaths,
    #     trainDataFilePaths=trainDataFilePaths, testDataFilePaths=testDataFilePaths,
    # ).allocate()
    CNNTrainer.fit(argv)
