# coding=utf-8
import math
import os

from src.util import common


def split_by_proportion(src_file_path, target_dir_path, split_file_cnt):
    """按照相等的概率(拟合频率)将文件 src_file 划分为 split_file_cnt 个文件, 并保存在 target_dir_path 目录下.
    不保证每个文件的行数严格相等, 只保证将 1 行分配到各文件的概率相等.
    """
    with open(src_file_path) as file:
        split_files = [open(os.path.join(target_dir_path, 'part-%d.split' % file_no), 'w')
                       for file_no in range(split_file_cnt)]
        for line in file:
            file_no = int(math.floor(common.new_proportion() * split_file_cnt))
            split_files[file_no].write(line)

        [split_file.close() for split_file in split_files]
