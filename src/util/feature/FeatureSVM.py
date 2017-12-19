#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pickle
import logging
import numpy as np
from src.util.BaseProcess import BaseProcess
from conf import FeatureSVMConf
from conf import DataType
from src.util.filter.InfoFilter import InfoFilter
from src.util.filter.RecordFilter import RecordFilter
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class UserBehaviorFeaturesCNN(BaseProcess):
    """
    用户行为特征提取
    """

    def __init__(self, user_type, input_data_path, output_data_path):
        BaseProcess.__init__(self, user_type, input_data_path, output_data_path)
        self.process_name = 'feature'
        self.content_type = 'svm'
        self.info_filter = None
        self.record_filter = None


    def process(self):
        """
        UserBehaviorFeatures类，函数执行流程
        :param file_path:
        :param data_type: fraud_user/normal_user
        :return:
        """
        if(self.info_filter == None or self.record_filter == None):
            raise "Class %s does not have filter." % self.__class__

        record_dataframe_dict = self.record_filter.get_data()

        data_X, data_y = self.trans_data(record_dataframe_dict)
        logging.info("Transfer to picture and save finished")
        return data_X, data_y


    def trans_data(self, record_dataframe_dict):

        #懒加载
        info_dict = pickle.load(open(self.info_filter.get_output_path(self.user_type + '_dict.pkl')))

        # 对train_x处理
        data_X = []
        data_y = []
        for key, record_df in record_dataframe_dict.items():  # key = fromnum, month_ind, day_ind, window_ind
            # 把行补全为60行
            reacord_arr = record_df.mean().tolist()

            #merge用户信息
            from_num = key.strip().split('_')[0]

            if(info_dict.has_key(from_num)):
                info = info_dict[from_num][FeatureSVMConf.COLUMNS_NAME]
            else:
                continue
            info_arr = info.tolist()
            feature = reacord_arr + info_arr
            data_X.append(feature)

            data_y.append(DataType[self.user_type].value)

            # 对列归一化
            # self.mkdirs(self.get_output_path(from_num + '.uid'))
            # with open(self.get_output_path(from_num + '.uid'), 'a') as data_x_file:
            #     data_x_file.write(data_x_str + '\n')
            #
            # data_y = FeatureCNNConf.LABEL_DICT[self.user_type]
            # with open(self.get_output_path(from_num + '.label'), 'a') as data_y_file:
            #     data_y_file.write(str(data_y) + '\n')

        data_X = np.asarray(data_X).reshape(-1, len(feature))
        data_y = np.asarray(data_y).reshape(-1, 1)
        return data_X, data_y

    def SVMInference(self, data_X, data_y):

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(data_X, data_y, test_size=0.2, random_state=7)
        clf = SVC(kernel='poly', degree=1, gamma=1, coef0=0)
        clf.fit(data_X, data_y)
        result = clf.predict(X_test)
        self.judge(result, y_test)


        SVC(C=1.0, kernel='rbf', degree=4, gamma='auto', coef0=0.0, shrinking=True, probability=False,
            tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None,
            random_state=None)


    def RFInference(self, data_X, data_y):

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(data_X, data_y, test_size=0.2,                                                                             random_state=7)
        clf = RandomForestClassifier(
             bootstrap=True, class_weight=None, criterion='gini',
             max_depth=50, max_features='auto', max_leaf_nodes=None, min_impurity_split=0,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
             oob_score=False, random_state=0, verbose=0, warm_start=False)
        clf.fit(X_train, y_train)
        result = clf.predict(X_test)
        self.judge(result, y_test)

    def judge(self, result, y_test):
        right_counts = 0
        for i in range(len(result)):
            if(result[i] == y_test[i]):
                right_counts += 1
        logging.info('%4f'%(1.0 * right_counts / len(result)))

    def set_info_filter(self, info_filter):
        self.info_filter = info_filter

    def set_record_filter(self, record_filter):
        self.record_filter = record_filter


if __name__ == '__main__':

    input_dir = '/Users/mayuchen/Documents/Python/Repository/DL/Other/UserBehaviorMining/resource/little_data/'
    output_dir = '/Users/mayuchen/Documents/Python/Repository/DL/Other/UserBehaviorMining/resource/data/'


    normal_info = InfoFilter(DataType.normal.name, input_dir + '/info/普通用户号码_md5.xlsx', output_dir)
    normal_record = RecordFilter(DataType.normal.name, input_dir + '/normal_user/', output_dir)
    normal_users = UserBehaviorFeaturesCNN(DataType.normal.name, input_dir, output_dir)
    normal_users.set_info_filter(normal_info)
    normal_users.set_record_filter(normal_record)
    normal_data_X, normal_data_y = normal_users.process()

    fraud_info = InfoFilter(DataType.fraud.name, input_dir + '/info/list360诈骗电话-0512_md5.xlsx', output_dir)
    # fraud_info.process()
    fraud_record = RecordFilter(DataType.fraud.name, input_dir + '/fraud_user/', output_dir)
    fraud_record.process()
    fraud_users = UserBehaviorFeaturesCNN(DataType.fraud.name, input_dir, output_dir)
    fraud_users.set_info_filter(fraud_info)
    fraud_users.set_record_filter(fraud_record)
    fraud_data_X, fraud_data_y = fraud_users.process()

    normal_X = pickle.load(open('./normal_X.pkl')).tolist()
    normal_y = pickle.load(open('./normal_y.pkl')).tolist()
    fraud_X = pickle.load(open('./fraud_X.pkl')).tolist()
    fraud_y = pickle.load(open('./fraud_y.pkl')).tolist()

    data_X = normal_X
    data_y = normal_y

    normal_X.extend(fraud_X)
    normal_y.extend(fraud_y)

    data_X = np.nan_to_num(data_X)
    test_users = UserBehaviorFeaturesCNN(None, None, None)
    test_users.RFInference(np.asarray(data_X), np.asarray(data_y))


