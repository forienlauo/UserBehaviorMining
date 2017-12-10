#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pandas as pd
import logging
import random
import os
from scipy import misc
import sklearn.preprocessing
from src.util.BaseProcess import BaseProcess
from conf import FeatureCNNConf
from conf import DataType
from src.util.filter.InfoFilter import InfoFilter
from src.util.filter.RecordFilter import RecordFilter



class FeaturesCNN(BaseProcess):
    """
    用户行为特征提取
    """

    def __init__(self, user_type, output_data_path):
        BaseProcess.__init__(self, user_type, output_data_path)
        self.process_name = 'feature'
        self.content_type = 'cnn'
        self.info_filter = None
        self.record_filter = None


    def process(self):
        """
        UserBehaviorFeatures类，函数执行流程
        :param file_path:
        :param data_type: fraud_user/normal_user
        :return:
        """
        if(self.record_filter == None):
            raise "Class %s does not have filter." % self.__class__

        record_dataframe_dict = self.record_filter.get_data()

        self.trans_image(record_dataframe_dict)

        return self


    def trans_image(self, record_dataframe_dict):

        # 对train_x处理
        min_max_scaler = sklearn.preprocessing.MinMaxScaler()

        for key, record_df in record_dataframe_dict.items():  # key = fromnum, month_ind, day_ind, window_ind
            # 把行补全为60行
            index = pd.DataFrame({'index': pd.Series(range(60), index=range(60))})
            ind_record_df = pd.merge(index, record_df, left_index=True, right_index=True, how='left')

            ind_record_df.drop(['index'], axis=1, inplace=True)

            # 补全缺失值
            ind_record_df.interpolate(inplace=True)
            ind_record_df.fillna(method='bfill', inplace=True)
            # 若全部为空，则
            if ind_record_df.dropna(how='all').empty:
                continue
            ind_record_df.fillna(0, inplace=True)

            #merge用户信息
            from_num = key.strip().split('_')[0]

            # if(info_dict.has_key(from_num)):
            #     info = info_dict[from_num]
            # else:
            #     continue

            #将info信息转成dataframe格式
            # info_cols = {}
            # for (k, val) in dict(info).items():
            #     info_cols[k] = pd.Series(60 * [val])
            # info_df = pd.DataFrame(info_cols)

            #record和info的merge
            # ind_record_info_df = pd.merge(ind_record_df, info_df, left_index=True, right_index=True)

            columns_name_list = []
            columes_list = FeatureCNNConf.COLUMNS_NAME

            for i in range(FeatureCNNConf.REPETITION_COUNTS):
                random.seed(7 * i)
                random.shuffle(columes_list)
                columns_name_list.extend(columes_list)

            data_df = pd.DataFrame([])
            for i, name in enumerate(columns_name_list):
                data_df.insert(0, '%s_%s' % (name, i), ind_record_df[name])

            # 对列归一化
            data_arr = min_max_scaler.fit_transform(data_df)
            columes_count = FeatureCNNConf.REPETITION_COUNTS * len(FeatureCNNConf.COLUMNS_NAME)
            data_pic = misc.imresize(data_arr.reshape(60, columes_count), 1.0)
            data_x_str = data_pic.reshape(1, -11).astype(str).tolist()[0]
            data_x_str = ','.join(data_x_str)

            self.mkdirs(os.path.dirname(self.get_output_path(from_num + '.uid')))
            with open(self.get_output_path(from_num + '.uid'), 'a') as data_x_file:
                data_x_file.write(data_x_str + '\n')

            data_y = DataType[self.user_type].value
            with open(self.get_output_path(from_num + '.label'), 'a') as data_y_file:
                data_y_file.write(str(data_y) + '\n')

        logging.info("Transfer to picture and save finished")


    def set_info_filter(self, info_filter):
        self.info_filter = info_filter

    def set_record_filter(self, record_filter):
        self.record_filter = record_filter


if __name__ == '__main__':

    info = InfoFilter(DataType.normal.name,
                      '/Users/mayuchen/Documents/Python/Repository/DL/Other/UserBehaviorMining/resource/raw/info/普通用户号码_md5.xlsx',
                      '/Users/mayuchen/Documents/Python/Repositorygr/DL/Other/UserBehaviorMining/resource/data/')

    record = RecordFilter(DataType.normal.name,
                          '/Users/mayuchen/Documents/Python/Repository/DL/Other/UserBehaviorMining/resource/raw/normal_user',
                          '/Users/mayuchen/Documents/Python/Repository/DL/Other/UserBehaviorMining/resource/data/')


    users = FeaturesCNN(DataType.normal.name, '/Users/mayuchen/Documents/Python/Repository/DL/Other/UserBehaviorMining/resource/data/', '/Users/mayuchen/Documents/Python/Repository/DL/Other/UserBehaviorMining/resource/data/')
    users.set_info_filter(info)
    users.set_record_filter(record)
    users.process()



