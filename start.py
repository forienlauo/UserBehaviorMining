#!/usr/bin/env python
# coding=utf-8
import sys

import conf
from src.predict import CNNPredictor
from src.prepare import Prepare
from src.train import CNNTrainer


def print_usage():
    print("""
need args:
    <module_name> [module_argv]
where
    module_name is <prepare|train|predict>
    module_argv depends on concrete module

a common interface for prepare, train and predict modules.
predict module is experimental yet.
""")


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print_usage()
        sys.exit(1)

    conf.init()

    module_name = sys.argv[1]
    module_argv = sys.argv[2:]
    exit_code = 0
    if module_name == 'prepare':
        Prepare.prepare(module_argv)
    elif module_name == 'train':
        exit_code = CNNTrainer.fit(module_argv)
    elif module_name == 'predict':
        exit_code = CNNPredictor.fit(module_argv)
    else:
        print_usage()
        exit_code = 1

    sys.exit(exit_code)
