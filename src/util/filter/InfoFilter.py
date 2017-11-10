#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import datetime
import os
import pickle
import time

import numpy as np
import pandas as pd

from conf import DataType
from conf import InfoConf

from src.util.BaseProcess import BaseProcess


class InfoFilter(BaseProcess):

    def __init__(self, user_type, input_data_path, output_data_path):
        BaseProcess.__init__(self, user_type, output_data_path)
        self.process_name = 'filter'
        self.content_type = 'info'
        self.input_data_path = input_data_path

    def process(self):
        dump_path_dict = self.get_output_path("%s_dict.pkl"%(self.user_type))
        dump_path_in = self.get_output_path("%s.pkl"%(self.user_type))
        dump_path_ex = self.get_output_path("%s.pkl"%(self.user_type), 'exclude')
        self.mkdirs(dump_path_in)
        self.mkdirs(dump_path_ex)
        if os.path.exists(dump_path_dict):
            user_info_dict = pickle.load(open(dump_path_dict))
        else:
            user_info_df = pd.read_excel(self.input_data_path)
            user_info_df_include, user_info_df_exclude, user_info_dict = self.check_data(user_info_df, self.user_type)
            pickle.dump(user_info_dict, open(dump_path_dict, 'w'))
            # pickle.dump(user_info_df_include, open(dump_path_in, 'w'))
            # pickle.dump(user_info_df_exclude, open(dump_path_ex, 'w'))

            user_info_df_include.to_csv(dump_path_in + '.csv', index = True, header = True)
            user_info_df_exclude.to_csv(dump_path_ex + '.csv', index = True, header = True)
        return self


    def check_data(self, user_info_df, user_type):

        def parse_time_normal(tm):
            try:
                dtime = datetime.datetime.strptime(str(tm), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return np.nan
            else:
                return int((time.mktime(datetime.datetime.now().timetuple()) - time.mktime(dtime.timetuple())) / 864000)

        def parse_time_fraud(tm):
            '''
            距离当前时间的天数
            '''
            try:
                dtime = datetime.datetime.strptime(str(tm), "%Y%m%d000000")
            except ValueError:
                return np.nan
            else:
                return int((time.mktime(datetime.datetime.now().timetuple()) - time.mktime(dtime.timetuple())) / 86400)

        def parse_sell_product(sp):
            if InfoConf.SELL_PRODUCT_DICT.has_key(sp):
                return InfoConf.SELL_PRODUCT_DICT[sp]
            elif (isinstance(sp, unicode) and sp.encode('utf-8') != '\N'):
                return InfoConf.SELL_PRODUCT_DICT[u'其他']
            else:
                return np.nan

        def do_clean_train(row):
            row = row.values.astype(str)
            for item in row:
                if len(item) == 0 or item == 'nan' or item == '\N':
                    return False
            return True

        def parse_plan_name(plan_name):
            if(type(plan_name) == float and str(plan_name) == 'nan'):
                return 0
            plan_name = plan_name.encode("utf-8")
            if(str(plan_name)[:4].isdigit()):
                year = int(str(plan_name)[:4])
                if(year < 2018 and year >= 2008):
                    return (year - 2009) * 20
                else:
                    return 0
                return int(str(plan_name)[:4])
            return 0


        #只取部分列
        user_info_df = user_info_df[InfoConf.INFO_TABLE_IND.keys()]

        user_info_df = user_info_df.rename(columns=InfoConf.INFO_TABLE_IND)

        user_info_df['from_num'] = user_info_df['from_num'].apply(lambda num: num if len(num)>=30 and len(num)<=32 else np.nan)

        user_info_df['user_type'] = user_info_df['user_type'].apply(lambda type: InfoConf.INFO_TYPE_DICT[type] if InfoConf.INFO_TYPE_DICT.has_key(type) else np.nan)

        if(user_type == DataType.normal.name or user_type == DataType.test.name):
            user_info_df['open_tate'] = user_info_df['open_tate'].apply(parse_time_normal)
        else:
            user_info_df['open_tate'] = user_info_df['open_tate'].apply(parse_time_fraud)

        user_info_df['sub_stattp'] = user_info_df['sub_stattp'].apply(lambda stat: InfoConf.INFO_SUB_STAT_DICT[stat] if InfoConf.INFO_SUB_STAT_DICT.has_key(stat) else np.nan)

        user_info_df['is_realname'] = user_info_df['is_realname'].apply(lambda is_real: 200 if is_real == 1 else 0)

        user_info_df['sell_product'] = user_info_df['sell_product'].apply(parse_sell_product)

        user_info_df['plan_name'] = user_info_df['plan_name'].apply(parse_plan_name)

        if(user_type == DataType.test.name):
            user_info_df = user_info_df.fillna(0)
            user_info_df['is_include'] = True
        else:
            user_info_df['is_include'] = user_info_df.apply(do_clean_train, axis=1)

        user_info_df_include = user_info_df[user_info_df.is_include == True]
        user_info_df_exclude = user_info_df[user_info_df.is_include == False]

        user_info_df_include = user_info_df_include.drop_duplicates('from_num')

        user_info_dict = self.trans_dict(user_info_df_include)

        return user_info_df_include, user_info_df_exclude, user_info_dict

    def trans_dict(self, user_info_df_include):
        user_info_dict = {}
        def make_dict(row):
            user_info_dict[row['from_num']] = row.copy()
        user_info_df_include.apply(make_dict, axis=1)
        return user_info_dict


if __name__ == '__main__':
    # info = InfoFilter(DataType.normal.name, '/Users/mayuchen/Documents/Python/Repository/DL/Other/UserBehaviorMining/resource/raw/info/普通用户号码_md5.xlsx', '/Users/mayuchen/Documents/Python/Repository/DL/Other/UserBehaviorMining/resource/data/')
    # info.process()

    info = InfoFilter(DataType.fraud.name, '/Users/mayuchen/Documents/Python/Repository/DL/Other/UserBehaviorMining/resource/raw/info/list360诈骗电话-0512_md5.xlsx', '/Users/mayuchen/Documents/Python/Repository/DL/Other/UserBehaviorMining/resource/data/')
    info.process()
