#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys

sys.path.append('../../')

import pandas as pd
import numpy as np
import pickle
import os
import conf

SEPARATOR = '|'
FEATURE_NUM = 11
PARTITION_COUNT = 20

columns_dict = dict(
    from_num=[0, int],
    to_num=[1, int],
    charge_num=[2, int],
    start_time=[3, int],
    duration=[4, int],
    cost=[5, int],
    type=[6, int],
    call_type=[7, int],
    talk_type=[8, int],
    from_area=[9, int],
    to_area=[10, int],
)


def partitionProcess(data_dir, user_type):
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for file_name in filenames:
            if not file_name.endswith('txt.md5'):
                continue
            partition(dirpath + '/' + file_name, user_type)
    return


def partition(file_dir, user_type):
    """
    以主叫号码为key做partition，便于数据处理
    :param file_dir:
    :param file_dir: 用户类型，(正常用户、诈骗用户)
    :return:
    """
    basename = os.path.basename(file_dir)

    with open(file_dir, "r") as file:
        lines = file.readlines()
    exclude_lines = []
    partition_dict = {}
    for line in lines:
        if check_data_line(line):
            part_key = partition_hash(line)
            if not partition_dict.has_key(part_key):
                partition_dict[part_key] = []
            partition_dict[part_key].append(line)
        else:
            exclude_lines.append(line)

    # 被过滤掉的数据
    if len(exclude_lines) != 0:
        exclude_file_path = __makepath("%s_%s.txt" % (user_type, basename), 'exclude')
        with open(exclude_file_path, 'w') as exclude_lines_file:
            exclude_lines_file.write(''.join(exclude_lines))

    # 干净的数据
    columns = sorted(columns_dict.iteritems(), key=lambda dict: dict[1])  # 对字典的value进行排序
    columns_name = [name for name, num in columns]
    for (part_key, include_lines) in partition_dict.items():
        include_file_path = __makepath("%s_%s.txt" % (user_type, part_key), 'include')
        with open(include_file_path, 'a') as include_lines_file:
            if include_lines_file.tell() == 0:  # 若写入偏移量为零，则将列名写入
                include_lines_file.write('|'.join(columns_name) + '\n')
            include_lines_file.write(''.join(include_lines))

    print user_type, basename + ' is finished.'


def __makepath(basename, clude_type):
    """
    文件路径
    :param basename:   原文件名
    :param clude_type:    数据类型(干净的数据、 滤掉的数据)
    :return:
    """
    return os.path.join(conf.ROOT_DIR, "resource/filtered/%s" % (clude_type), basename)


def partition_hash(line):
    '''
    partition映射规则： 主叫号码后三位转为10进制求余
    :param line: 一条数据
    :return: 映射值
    '''
    line = line.strip()
    from_num = [feature.strip() for feature in line.split(SEPARATOR)][0]
    if (len(from_num) == 0):
        return
    return int(from_num[-3:], 16) % PARTITION_COUNT


def check_data_line(line):
    # configuration
    line = line.strip()
    row = [feature.strip() for feature in line.split(SEPARATOR)]
    if len(row) != FEATURE_NUM:
        return False

    from_num = row[columns_dict['from_num'][0]]
    if not (len(from_num) <= 32 or len(from_num) >= 30):
        return False

    to_num = row[columns_dict['to_num'][0]]
    if not (len(to_num) <= 32 or len(to_num) >= 30):
        return False

    charge_num = row[columns_dict['charge_num'][0]]
    if not (len(charge_num) <= 32 or len(charge_num) >= 30):
        return False

    start_time = row[columns_dict['start_time'][0]]
    if not (len(start_time) == 14 and start_time.isdigit()):
        return False

    duration = row[columns_dict['duration'][0]]
    if not (duration.isdigit() and duration > 0):
        return False

    cost = row[columns_dict['cost'][0]]
    if not (cost.isdigit() and cost >= 0):
        return False

    type = row[columns_dict['type'][0]]
    if not len(type) > 0:
        return False
    if type == 'local' or type == 'distance':  # 如果通话类型为本地、外地固话，则可能没有后面的数据
        return True

    call_type = row[columns_dict['call_type'][0]]
    if not len(call_type) > 0:
        return False

    talk_type = row[columns_dict['talk_type'][0]]
    if not len(talk_type) > 0:
        return False

    from_area = row[columns_dict['from_area'][0]]
    if not (from_area.isdigit() and from_area > 0):
        return False

    to_area = row[columns_dict['to_area'][0]]
    if not (to_area.isdigit() and to_area > 0):
        return False

    return True


if __name__ == '__main__':
    partitionProcess('../../resource/raw/fraud_user', 'fraud_user')
    partitionProcess('../../resource/raw/normal_user', 'normal_user')
