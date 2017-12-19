# coding=utf-8
import math
import os
import logging
import sys
from conf import DataType
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


def split_fraud_normal(record_data_path, output_data_path):

    split_symbol = '\t'

    fraud_dict = {}
    with open('./configuration/fraud_num.txt', 'r') as fraud_num_file:
        fraud_num_lines = fraud_num_file.readlines()
        for line in fraud_num_lines:
            from_num = line.strip()
            fraud_dict[from_num] = True

    # 解析record_data
    fraud_txt = []
    normal_txt = []
    record_counts = [0, 0]
    for record_file_dir in os.listdir(record_data_path):
        reacord_name = os.path.basename(record_file_dir)
        if not (reacord_name.endswith('.out') or reacord_name.endswith('.txt')):
            continue
        with open(os.path.join(record_data_path, record_file_dir), 'r') as record_file:
            record_lines = record_file.readlines()
            for line in record_lines:
                from_num = line.split(split_symbol)[0].strip()
                if(fraud_dict.has_key(from_num)):
                    fraud_txt.append(line)
                    record_counts[DataType.fraud.value] += 1
                else:
                    normal_txt.append(line)
                    record_counts[DataType.normal.value] += 1

    if(record_counts[DataType.fraud.value] == 0 or record_counts[DataType.normal.value] == 0):
        logging.error('No fraud or normal data !')
        sys.exit(1)

    logging.info("fraud record cnt: %s" % record_counts[DataType.fraud.value])
    logging.info("normal record cnt: %s" % record_counts[DataType.normal.value])


    #存储record_data中fraud部分
    fraud_dir = '%s/raw_data/fraud'%(output_data_path)
    if not os.path.exists(fraud_dir):
        os.makedirs(fraud_dir)
    fraud_file_path = '%s/fraud.txt.md5'%(fraud_dir)
    with open(fraud_file_path, 'w') as fraud_file:
        fraud_file.writelines(fraud_txt)

    # 存储record_data中normal部分
    normal_dir = '%s/raw_data/normal/'%(output_data_path)
    if not os.path.exists(normal_dir):
        os.makedirs(normal_dir)
    normal_file_path = '%s/normal.txt.md5' % (normal_dir)
    with open(normal_file_path, 'w') as normal_file:
        normal_file.writelines(normal_txt)

    return fraud_file_path, normal_file_path





