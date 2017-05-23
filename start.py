import sys

import conf
import train

if __name__ == '__main__':
    sys.argv.extend([
        # train_data
        'tmp/cnn_test/train_data.in',
        # test_data
        'tmp/cnn_test/test_data.in',
        # initial_height
        str(5 * 60),
        # initial_width
        str(10),
        # target_class_cnt
        str(2),
    ])

    conf.init()
    train.cnn(sys.argv)
