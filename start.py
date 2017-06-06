# coding=utf-8
import sys

import conf
import train

if __name__ == '__main__':
    sys.argv.extend([
        # cnn argv
        # test
        '\t',
        'tmp/cnn_test/train_data_x.in', 'tmp/cnn_test/train_data_y.in',
        'tmp/cnn_test/test_data_x.in', 'tmp/cnn_test/test_data_y.in',
        5 * 60, 20, 1, 2,
        200, 10,
    ])
    argv = map(lambda x: str(x), sys.argv)

    conf.init()

    train.cnn(argv)
