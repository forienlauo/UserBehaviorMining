#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
sys.path.append('../../')

import os
import pandas as pd
import numpy as np
import random
import sklearn.preprocessing
from scipy import misc

class UserBehaviorFeatures():
    """
    用户行为特征提取
    """
    def __init__(self):
        self.interval = 5   #5分钟为采样间隔
        self.sampling_month = 3  # 只取三月数据
        self.sampling_hour = [(7, 12), (12, 17), (17, 22)]  # 每天采样的时间段，单位为小时
        self.dump_path = '../../resource/merge/'
        self.columns_name = []

    def extrace_and_process(self ,file_path, data_type):
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
                    feature.append(pd.DataFrame(getattr(self, funs)(dataframe ,funs)))
                feature_dict[key] = pd.concat(feature, axis=1)
            train_x, train_y = self.after_merge(feature_dict, data_type)

            print("extracing finished")
        train_x = pd.DataFrame(train_x.reshape(-1, 60*36))
        train_y = pd.DataFrame(train_y.reshape(-1, 2))
        train_x.to_csv(self.dump_path + basename + '_train_x.txt', index=False, header=False, sep=',')
        train_y.to_csv(self.dump_path + basename + '_train_y.txt', index=False, header=False, sep=',')
        return train_x, train_y

    def get_data(self, path):
        data = None
        if(os.path.isdir(path)):
            file_list = os.listdir(path)  # 列出目录下的所有文件和目录
            for file_name in file_list:
                filepath = os.path.join(path, file_name)
                if data is None:
                    data = pd.read_csv(filepath, sep='|')
                else:
                    data_tmp = pd.read_csv(filepath, sep='|')
                    data = pd.concat([data, data_tmp])
        elif(os.path.isfile(path)):      #输入若是文件名，则是小数量样本集
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

        #type类型映射
        type_map = {'cvoi':0, 'local':1, 'distance':2}
        data['type'] = data['type'].apply(lambda x: type_map[x])

        data = data[data.month == self.sampling_month]

        data_dict = {}
        for fromnum in set(data.from_num):
            print fromnum
            for day_ind in range(1, 32):
                for window_ind in range(3):
                    key = '%s_%s_%s' %(fromnum, day_ind, window_ind)
                    data_tmp = data[(data.from_num == fromnum) & (data.day == day_ind)]
                    data_tmp = data_tmp[(data_tmp.hour >= self.sampling_hour[window_ind][0]) & (data_tmp.hour < self.sampling_hour[window_ind][1])]
                    if not data_tmp.empty:
                        data_dict[key] = data_tmp
                    data_tmp['window_ind'] = window_ind
                    data_tmp['interval_in_window'] = (((data_tmp.hour - self.sampling_hour[window_ind][0]) * 60 + data_tmp.minute) / self.interval).astype(int)

        return data_dict


    def after_merge(self, feature_dict, data_type):
        #图片数量
        pic_num = len(feature_dict.keys())

        #对train_x处理
        min_max_scaler = sklearn.preprocessing.MinMaxScaler()
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

            columes_list = list(dataframe.columns)
            #由于列数较少，所以打乱顺序复制
            # if len(self.columns_name) == 0:
            #     for i in range(4):
            #         random.shuffle(columes_list)
            #         self.columns_name.extend(columes_list)
            #     print columes_list, self.columns_name
            #列序固定
            self.columns_name = ['fe_all_call_count_count', 'fe_cost_mean', 'fe_cost_std', 'fe_duration_mean', 'fe_duration_std',
             'fe_type_median',
             'fe_all_call_count_count', 'fe_duration_std', 'fe_duration_mean', 'fe_cost_std', 'fe_cost_mean',
             'fe_type_median',
             'fe_all_call_count_count', 'fe_duration_std', 'fe_cost_std', 'fe_duration_mean', 'fe_cost_mean',
             'fe_type_median',
             'fe_all_call_count_count', 'fe_type_median', 'fe_cost_std', 'fe_duration_mean', 'fe_cost_mean',
             'fe_duration_std',
             'fe_duration_std', 'fe_all_call_count_count', 'fe_type_median', 'fe_duration_mean', 'fe_cost_std',
             'fe_cost_mean']

            for i, name in enumerate(self.columns_name):
                dataframe.insert(0, '%s_%s' % (name, i), dataframe[name])
            # 对列归一化
            data_arr = min_max_scaler.fit_transform(dataframe)

            train_x_list.append(misc.imresize(data_arr, (60, 36)))  # 图片转成64X64

        train_x = np.asarray(train_x_list, dtype=np.uint8).reshape(pic_num, 60, 36)
        # train_x = np.asarray(train_x_list, dtype=np.uint8).reshape(pic_num, 64, 64)

        # 对train_y处理
        train_y_list = []
        train_y_res = [0, 1] if data_type == 'fraud_user' else [1, 0]
        for i in range(pic_num):
            train_y_list.append(train_y_res)
        train_y = np.asarray(train_y_list, dtype=np.uint8).reshape(pic_num, 2)

        return train_x, train_y

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
    train_x, train_y = users.extrace_and_process('../../resource/little_data/filtered/include/fraud_user_0_sample.txt', 'fraud_user')
    train_x, train_y = users.extrace_and_process('../../resource/little_data/filtered/include/normal_user_0.txt', 'normal_user')


