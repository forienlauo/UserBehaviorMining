#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys

import os
import pandas as pd
import numpy as np
import random
import pickle
from scipy import misc
from enum import Enum, unique
import sklearn.preprocessing

@unique
class DataType(Enum):
    NORMAL = 'normal'
    FRAUD = 'fraud'
    TEST = 'test'

class UserBehaviorFeatures():
    """
    用户行为特征提取
    """

    def __init__(self):
        self.interval = 5  # 5分钟为采样间隔
        self.sampling_month = (3, 5)                        #采样月份
        self.sampling_hour = [(7, 12), (12, 17), (17, 22)]  # 每天采样的时间段，单位为小时
        self.dump_path = '../../resource/little_data/merge/'
        self.columns_name = []
        #self.fraud_info_dict = pickle.load(open('resource/little_data/filtered/info/include/fraud_dict.pkl'))
        #self.normal_info_dict = pickle.load(open('resource/little_data/filtered/info/include/normal_dict.pkl'))
        self.test_info_dict = pickle.load(open('../../resource/little_data/filtered/info/include/test_dict.pkl'))

    def extrace_and_process(self, file_path, data_type):
        """
        UserBehaviorFeatures类，函数执行流程
        :param file_path:
        :param data_type: fraud_user/normal_user
        :return:
        """
        basename = os.path.basename(file_path)
        if os.path.exists(self.dump_path + basename + '_train_x.npy'):
            train_x = np.load(self.dump_path + basename + '_train_x.npy')
            train_y = np.load(self.dump_path + basename + '_train_y.npy')
        else:
            train_x = self.get_data(file_path)
            data_dict = self.before_merge(train_x)
            feature_dict = {}
            for key, dataframe in data_dict.items():
                feature = []
                for funs in (i for i in dir(self) if i[:3] == 'fe_'):
                    feature.append(pd.DataFrame(getattr(self, funs)(dataframe, funs)))
                feature_dict[key] = pd.concat(feature, axis=1)
        print("extracing finished")

        if not data_type == DataType.TEST.value:
            train_x, train_y = self.after_merge(feature_dict, data_type)
            train_x = pd.DataFrame(train_x.reshape(-1, 60 * 88))
            train_y = pd.DataFrame(train_y.reshape(-1, 2))
            train_x.to_csv(self.dump_path + basename + '_train_x.txt', index=False, header=False, sep=',')
            train_y.to_csv(self.dump_path + basename + '_train_y.txt', index=False, header=False, sep=',')
            return train_x, train_y
        else:
            test_dict = self.after_merge(feature_dict, data_type)
            for from_num in test_dict:
                test_x = pd.DataFrame(test_dict[from_num].reshape(-1, 60 * 88))
                test_x.to_csv(self.dump_path + 'test/' + from_num + '.txt', index=False, header=False, sep=',')
            return train_x, train_y

    def get_data(self, path):
        data = None
        if (os.path.isdir(path)):
            file_list = os.listdir(path)  # 列出目录下的所有文件和目录
            for file_name in file_list:
                filepath = os.path.join(path, file_name)
                if data is None:
                    data = pd.read_csv(filepath, sep='|')
                else:
                    data_tmp = pd.read_csv(filepath, sep='|')
                    data = pd.concat([data, data_tmp])
        elif (os.path.isfile(path)):  # 输入若是文件名，则是小数量样本集
            data = pd.read_csv(path, sep='|')
        else:
            print '########## wrong. ##########'

        return data

    def before_merge(self, data):
        data['date'] = (data.start_time / 1000000).astype(int)
        data['year'] = (data.date / 10000).astype(int)
        data['month'] = ((data.date / 100) % 100).astype(int)
        data['day'] = (data.date % 100).astype(int)
        data['hour'] = ((data.start_time % 1000000) / 10000).astype(int)
        data['minute'] = ((data.start_time % 10000) / 100).astype(int)

        # type类型映射
        type_map = {'cvoi': 0, 'local': 1, 'distance': 2}
        data['type'] = data['type'].apply(lambda x: type_map[x])

        data_dict = {}
        sampling_month = self.sampling_month
        for fromnum in set(data.from_num):
            print fromnum
            for month_ind in range(sampling_month[0], sampling_month[1]):
                for day_ind in range(1, 32):
                    for window_ind in range(3):
                        key = '%s_%s_%s_%s' % (fromnum, month_ind, day_ind, window_ind)
                        data_tmp = data[(data.from_num == fromnum) & (data.month == month_ind) & (data.day == day_ind)]
                        data_tmp = data_tmp[(data_tmp.hour >= self.sampling_hour[window_ind][0]) & (
                            data_tmp.hour < self.sampling_hour[window_ind][1])]
                        if not data_tmp.empty:
                            data_dict[key] = data_tmp
                        data_tmp['window_ind'] = window_ind
                        data_tmp['interval_in_window'] = (((data_tmp.hour - self.sampling_hour[window_ind][
                            0]) * 60 + data_tmp.minute) / self.interval).astype(int)

        return data_dict

    def after_merge(self, feature_dict, data_type):
        # 图片数量
        #pic_num = len(feature_dict.keys())

        # 对train_x处理
        min_max_scaler = sklearn.preprocessing.MinMaxScaler()
        test_res_dict = {}
        train_x_list = []
        for key, dataframe in feature_dict.items():
            dataframe = feature_dict[key]

            # 把行补全为60行
            index = pd.DataFrame({'index': pd.Series(range(60), index=range(60))})
            dataframe = pd.merge(index, dataframe, left_index=True, right_index=True, how='left')

            dataframe.drop(['index'], axis=1, inplace=True)

            # 补全缺失值
            dataframe.interpolate(inplace=True)
            dataframe.fillna(method='bfill', inplace=True)

            # 若全部为空，则
            if dataframe.dropna(how='all').empty:
                continue
            dataframe.fillna(0, inplace=True)

            #merge用户信息
            from_num = key.strip().split('_')[0]
            if data_type == DataType.FRAUD.value:
                if(self.fraud_info_dict.has_key(from_num)):
                    info = self.fraud_info_dict[from_num]
                else:
                    continue
            elif data_type == DataType.NORMAL.value:
                if(self.normal_info_dict.has_key(from_num)):
                    info = self.normal_info_dict[from_num]
                else:
                    continue
            elif data_type == DataType.TEST.value:
                if (self.test_info_dict.has_key(from_num)):
                    info = self.test_info_dict[from_num]
                else:
                    continue

            info_dict = {}
            for (k, val) in dict(info).items():
                info_dict[k] = pd.Series(60 * [val])
            info_df = pd.DataFrame(info_dict)
            dataframe = pd.merge(dataframe, info_df, left_index=True, right_index=True)

            dataframe.drop(['from_num', 'is_include'], axis=1, inplace=True)

            columes_list = list(dataframe.columns)
            # 由于列数较少，所以打乱顺序复制
            if len(self.columns_name) == 0:
                for i in range(7):
                    random.shuffle(columes_list)
                    self.columns_name.extend(columes_list)
                print columes_list, self.columns_name
            # 列序固定
            self.columns_name = ['fe_type_median', 'open_tate', 'sub_stattp', 'fe_all_call_count_count', 'fe_cost_mean', 'user_type', 'fe_cost_std', 'fe_duration_std', 'fe_duration_mean', 'is_realname', 'sell_product', 'is_realname', 'fe_all_call_count_count', 'fe_duration_mean', 'sell_product', 'sub_stattp', 'fe_type_median', 'open_tate', 'fe_cost_mean', 'fe_cost_std', 'fe_duration_std', 'user_type', 'sub_stattp', 'fe_cost_mean', 'fe_duration_std', 'fe_type_median', 'fe_duration_mean', 'sell_product', 'fe_all_call_count_count', 'fe_cost_std', 'user_type', 'is_realname', 'open_tate', 'is_realname', 'user_type', 'fe_cost_mean', 'sub_stattp', 'fe_type_median', 'fe_all_call_count_count', 'fe_cost_std', 'fe_duration_mean', 'sell_product', 'open_tate', 'fe_duration_std', 'fe_duration_std', 'sub_stattp', 'open_tate', 'user_type', 'sell_product', 'is_realname', 'fe_cost_mean', 'fe_duration_mean', 'fe_type_median', 'fe_cost_std', 'fe_all_call_count_count', 'user_type', 'fe_cost_mean', 'fe_duration_std', 'fe_duration_mean', 'fe_all_call_count_count', 'sell_product', 'fe_type_median', 'is_realname', 'fe_cost_std', 'open_tate', 'sub_stattp', 'fe_type_median', 'fe_cost_std', 'sub_stattp', 'fe_duration_std', 'is_realname', 'fe_cost_mean', 'user_type', 'fe_duration_mean', 'fe_all_call_count_count', 'sell_product', 'open_tate']

            for i, name in enumerate(self.columns_name):
                dataframe.insert(0, '%s_%s' % (name, i), dataframe[name])

            # 对列归一化
            data_arr = min_max_scaler.fit_transform(dataframe)

            data_pic = misc.imresize(data_arr.reshape(60, 88), 1.0)

            if not data_type == DataType.TEST.value:
                train_x_list.append(data_pic)  # 图片转成64X88
            else:
                if not test_res_dict.has_key(from_num):
                    test_res_dict[from_num] = data_pic
                else:
                    test_res_dict[from_num] = np.append(test_res_dict[from_num], data_pic)

        # 对train_y处理
        if not data_type == DataType.TEST.value:
            train_x = np.asarray(train_x_list, dtype=np.uint8).reshape(len(train_x_list), 60, 88)
            train_y_list = []
            train_y_res = [0, 1] if data_type == DataType.FRAUD.value else [1, 0]
            for i in range(len(train_x_list)):
                train_y_list.append(train_y_res)
            train_y = np.asarray(train_y_list, dtype=np.uint8).reshape(len(train_x_list), 2)

            return train_x, train_y
        else:
            return test_res_dict

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


if __name__ == '__main__':
    users = UserBehaviorFeatures()
    #train_x, train_y = users.extrace_and_process('../../resource/little_data/filtered/record/include/fraud_user_0_sample.txt', DataType.FRAUD.value)
    #train_x, train_y = users.extrace_and_process('../../resource/little_data/filtered/record/include/normal_user_0.txt', DataType.NORMAL.value)
    users.extrace_and_process('../../resource/little_data/filtered/record/include/user_1.txt', DataType.TEST.value)


