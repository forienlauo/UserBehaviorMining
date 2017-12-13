#encoding=utf-8
import sys
import os
import glob
import pandas as pd
import logging
sys.path.append('../../../')

from conf import RecordConf
from conf import DataType

from src.util.BaseProcess import BaseProcess


class RecordFilter(BaseProcess):

    def __init__(self, user_type, input_data_path, output_data_path):
        BaseProcess.__init__(self, user_type, output_data_path)
        self.process_name = 'filter'
        self.content_type = 'record'
        self.input_data_path = input_data_path

    def process(self):
        output_path = os.path.join(self.output_data_path, self.process_name, self.content_type)
        if not os.path.exists(output_path):
            self.mkdirs(output_path)

        for file in os.listdir(output_path):
            if(file.startswith(self.user_type)):
                return self
        record_file_list = glob.glob(self.input_data_path + '/*txt.md5')
        if(len(record_file_list) == 0):
            logging.info('No record data has found, please check the data path')
            os._exit(0)
        for file_path in record_file_list:
            self.partition(file_path, self.user_type)
        return self


    def partition(self, file_dir, user_type):
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
            modify_line = self.check_data(line)
            if modify_line:
                part_key = self.partition_hash(line)
                if not partition_dict.has_key(part_key):
                    partition_dict[part_key] = []
                partition_dict[part_key].append(RecordConf.SEPARATOR.join(modify_line) + '\n')
            else:
                exclude_lines.append(line)

        # 被过滤掉的数据
        if len(exclude_lines) != 0:
            exclude_file_path = self.get_output_path("%s_%s.txt" % (user_type, basename), 'exclude')
            self.mkdirs(os.path.dirname(exclude_file_path))
            with open(exclude_file_path, 'w') as exclude_lines_file:
                exclude_lines_file.writelines(exclude_lines)

        # 干净的数据
        columns = sorted(RecordConf.RECORD_TABLE_IND.iteritems(), key=lambda dict: dict[1])  # 对字典的value进行排序
        columns_name = [name for name, num in columns]
        for (part_key, include_lines) in partition_dict.items():
            include_file_path = self.get_output_path("%s_%s.txt" % (user_type, part_key))
            self.mkdirs(os.path.dirname(include_file_path))
            with open(include_file_path, 'a') as include_lines_file:
                if include_lines_file.tell() == 0:  # 若写入偏移量为零，则将列名写入
                    include_lines_file.write(RecordConf.SEPARATOR.join(columns_name) + '\n')
                include_lines_file.writelines(include_lines)

        logging.info('%s(%s) is finished.'%(basename, user_type))


    def partition_hash(self, line):
        '''
        partition映射规则： 主叫号码后三位转为10进制求余
        :param line: 一条数据
        :return: 映射值
        '''
        line = line.strip()
        from_num = [feature.strip() for feature in line.split(RecordConf.SEPARATOR)][0]
        if (len(from_num) == 0):
            return
        return int(from_num[-3:], 16) % RecordConf.PARTITION_COUNT

    def get_data(self):
        output_path = self.get_output_path()
        record_dict = {}
        for file_path in glob.glob(output_path + self.user_type + '*.txt'):
            record = pd.read_csv(file_path, sep=RecordConf.SEPARATOR)
            record_dict = dict(record_dict, **(self.data_merge(record)))
            logging.info('%s has been gotten'%(file_path))

        feature_dict = {}
        for key, dataframe in record_dict.items():
            feature = []
            for funs in (i for i in dir(self) if i[:3] == 'fe_'):
                feature.append(pd.DataFrame(getattr(self, funs)(dataframe, funs)))
            feature_dict[key] = pd.concat(feature, axis=1)
        logging.info("Filter %s finished"%(self.user_type))
        return feature_dict

    def data_merge(self, data):
        '''
        取出原始数据，部分值并以时间粒度聚合
        :param data:
        :return:
        '''
        data['date'] = (data.start_time / 1000000).astype(int)
        data['year'] = (data.date / 10000).astype(int)
        data['month'] = ((data.date / 100) % 100).astype(int)
        data['day'] = (data.date % 100).astype(int)
        data['hour'] = ((data.start_time % 1000000) / 10000).astype(int)
        data['minute'] = ((data.start_time % 10000) / 100).astype(int)


        data_dict = {}
        for fromnum in set(data.from_num):
            for month_ind in range(RecordConf.SAMPLING_MONTH[0], RecordConf.SAMPLING_MONTH[1]):
                for day_ind in range(1, 32):
                    for window_ind in range(3):
                        key = '%s_%s_%s_%s' % (fromnum, month_ind, day_ind, window_ind)
                        data_tmp = data[(data.from_num == fromnum) & (data.month == month_ind) & (data.day == day_ind)]
                        data_tmp = data_tmp[(data_tmp.hour >= RecordConf.SAMPLING_HOUR[window_ind][0]) & (
                            data_tmp.hour < RecordConf.SAMPLING_HOUR[window_ind][1])]
                        data_tmp['window_ind'] = window_ind
                        data_tmp['interval_in_window'] = (((data_tmp.hour - RecordConf.SAMPLING_HOUR[window_ind][
                            0]) * 60 + data_tmp.minute) / RecordConf.INTERVAL).astype(int)
                        if not data_tmp.empty:
                            data_dict[key] = data_tmp
        return data_dict

    def fe_all_call_count(self, dataframe, feature_prefix):
        feature_dic = {}
        feature_name = "%s_count" % (feature_prefix)
        call_count = dataframe.groupby(['interval_in_window']).count()['to_num']
        feature_dic[feature_name] = call_count
        return feature_dic

    def fe_duration(self, dataframe, feature_prefix):
        feature_dic = {}
        feature_name_mean = "%s_mean" % (feature_prefix)
        feature_name_std = "%s_std" % (feature_prefix)
        group = dataframe.groupby(['interval_in_window'])
        feature_dic[feature_name_mean] = group.mean()['duration']
        feature_dic[feature_name_std] = group.std()['duration']
        return feature_dic

    def fe_cost(self, dataframe, feature_prefix):
        feature_dic = {}
        feature_name_mean = "%s_mean" % (feature_prefix)
        feature_name_std = "%s_std" % (feature_prefix)
        group = dataframe.groupby(['interval_in_window'])
        feature_dic[feature_name_mean] = group.mean()['cost']
        feature_dic[feature_name_std] = group.std()['cost']
        return feature_dic

    def fe_type(self, dataframe, feature_prefix):
        feature_dic = {}
        feature_name_median = "%s_median" % (feature_prefix)
        group = dataframe.groupby(['interval_in_window'])
        feature_dic[feature_name_median] = group.median()['type']
        return feature_dic


    def check_data(self, line):
        # configuration
        line = line.strip()
        row = [feature.strip() for feature in line.split(RecordConf.SEPARATOR)]
        if len(row) != RecordConf.FEATURE_NUM:
            return False

        from_num = row[RecordConf.RECORD_TABLE_IND['from_num'][0]]
        if not (len(from_num) <= 58 or len(from_num) >= 40):
            return False

        to_num = row[RecordConf.RECORD_TABLE_IND['to_num'][0]]
        if not (len(to_num) <= 58 or len(to_num) >= 40):
            return False

        charge_num = row[RecordConf.RECORD_TABLE_IND['charge_num'][0]]
        if not (len(charge_num) <= 58 or len(charge_num) >= 40):
            return False

        start_time = row[RecordConf.RECORD_TABLE_IND['start_time'][0]]
        if not (len(start_time) == 14 and start_time.isdigit()):
            return False

        duration = row[RecordConf.RECORD_TABLE_IND['duration'][0]]
        if not (duration.isdigit() and duration > 0):
            return False

        cost = row[RecordConf.RECORD_TABLE_IND['cost'][0]]
        if not (cost.isdigit() and cost >= 0):
            return False

        call_type = row[RecordConf.RECORD_TABLE_IND['call_type'][0]]
        if not len(call_type) > 0:
            return False

        longdis_type = row[RecordConf.RECORD_TABLE_IND['longdis_type'][0]]
        if not len(longdis_type) > 0:
            return False

        roaming_type = row[RecordConf.RECORD_TABLE_IND['roaming_type'][0]]
        if not len(roaming_type) > 0:
            return False

        return row


if __name__ == '__main__':
    #partitionProcess('resource/big_data/raw/test_user', 'user')
    record = RecordFilter(DataType.normal.name, '/Users/mayuchen/Documents/Python/Repository/DL/Other/UserBehaviorMining/resource/raw/normal_user', '/Users/mayuchen/Documents/Python/Repository/DL/Other/UserBehaviorMining/resource/data/')
    record.process()

    record = RecordFilter(DataType.normal.name, '/Users/mayuchen/Documents/Python/Repository/DL/Other/UserBehaviorMining/resource/raw/normal_user', '/Users/mayuchen/Documents/Python/Repository/DL/Other/UserBehaviorMining/resource/data/')
    record.process()
