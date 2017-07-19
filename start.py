# coding=utf-8
import sys

import conf
from src.train import CNNTrainer


def print_usage():
    print("""
need args:
    <module_name> [module_argv]
where
    module_name is <prepare|train>
    module_argv depends on concrete module
""")


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print_usage()
        sys.exit(1)

    conf.init()

    module_name = sys.argv[1]
    module_argv = sys.argv[1:]
    exit_code = 0
    if module_name == 'prepare':
        # 2017/07/19,niuqiang TODO call module prepare
        pass
    elif module_name == 'train':
        exit_code = CNNTrainer.fit(module_argv)
    else:
        print_usage()
        exit_code = 1

    sys.exit(exit_code)
