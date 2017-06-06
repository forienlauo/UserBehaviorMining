# coding=utf-8
import sys

import conf
import train
import util.example_utils as example

if __name__ == '__main__':
    generatedDataFilePaths = ('resource/little_data/data_x.txt', 'resource/little_data/data_y.txt',)
    trainDataFilePaths = ('resource/little_data/train_data_x.txt', 'resource/little_data/train_data_y.txt',)
    testDataFilePaths = ('resource/little_data/test_data_x.txt', 'resource/little_data/test_data_y.txt',)
    sys.argv.extend([
        # cnn argv
        # # test
        # '\t',
        # 'tmp/cnn_test/train_data_x.in', 'tmp/cnn_test/train_data_y.in',
        # 'tmp/cnn_test/test_data_x.in', 'tmp/cnn_test/test_data_y.in',
        # 5 * 60, 20, 1, 2,
        # 200, 10,
        # little data
        ',',
        trainDataFilePaths[0], trainDataFilePaths[1],
        testDataFilePaths[0], testDataFilePaths[1],
        60, 36, 1, 2,
        1000, 20,
    ])
    argv = map(lambda x: str(x), sys.argv)

    conf.init()

    example.ExampleAllocator(
        generatedDataFilePaths,
        trainDataFilePaths=trainDataFilePaths, testDataFilePaths=testDataFilePaths,
    ).allocate()
    train.cnn(argv)
