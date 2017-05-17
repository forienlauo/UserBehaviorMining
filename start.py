import sys

import conf
import train

if __name__ == '__main__':
    conf.init()
    train.cnn(sys.argv)
