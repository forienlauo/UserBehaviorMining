#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import os
import pandas as pd
import numpy as np
import time
import datetime
import pickle
from enum import Enum, unique


sys.path.append('../../')

@unique
class DataType(Enum):
    NORMAL = 'normal'
    FRAUD = 'fraud'
    TEST = 'test'

columns_dict = {
    u'主叫':'from_num',
    # u'用户名（userName）':'user_name',
    #u'产品名（planName）':'plan_name',
    #u'客户标识（custCode）':'cust_code',
    u'用户类别（userType）':'user_type',
    u'开户时间（openDate）':'open_tate',
    u'是否停机（subStatTp）':'sub_stattp',
    u'是否实名制（isRealname）':'is_realname',
    u'销售产品（sellProduct）':'sell_product',
    # u'360标注类型（telType）':'tel_type',
    # u'360标注次数（telCount）':'tel_count',
    # u'业务行为（xsub_action_cd）':'xsub_action_cd',
    # u'停机日期（xstop_dt）':'xstop_dt'
}

type_dict = {
    u'政企' : 0,
    u'公众' : 100
}

sub_stat_dict = {
    u'活动' : 0,
    u'停机' : 40,
    u'拆机' : 80,
    u'帐务停机': 120,
    u'割接' : 160
}

sell_product_dict = {
    u'CDMA预付费' : 0,
    u'CDMA后付费' : 30,
    u'CDMA准实时预付费' : 60,
    u'C+W（E+W）预付费' : 90,
    u'C+W（E+W）后付费' : 120,
    u'C+W（E+W）准实时预付费': 150,
    u'其他': 180
}


def filterProcess(info_data_dir, data_type):
    dump_path_dict = ''.join(['../../../resource/little_data/filtered/info/include/', data_type+'_dict', '.pkl'])
    dump_path_in = ''.join(['../../../resource/little_data/filtered/info/include/', data_type, '.pkl'])
    dump_path_ex = ''.join(['../../../resource/little_data/filtered/info/exclude/', data_type, '.pkl'])
    if os.path.exists(dump_path_dict):
        user_info_dict = pickle.load(open(dump_path_dict))
        #user_info_df_include = pickle.load(open(dump_path_in))
        #user_info_df_exclude = pickle.load(dump_path_ex)
    else:
        user_info_df = pd.read_excel(info_data_dir)
        user_info_df_include, user_info_df_exclude, user_info_dict = check_data(user_info_df, data_type)
        pickle.dump(user_info_dict, open(dump_path_dict, 'w'))
        pickle.dump(user_info_df_include, open(dump_path_in, 'w'))
        pickle.dump(user_info_df_exclude, open(dump_path_ex, 'w'))

    return user_info_dict


def check_data(user_info_df, data_type):

    def parse_time_normal(tm):
        try:
            dtime = datetime.datetime.strptime(str(tm), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return np.nan
        else:
            return int((time.mktime(datetime.datetime.now().timetuple()) - time.mktime(dtime.timetuple())) / 86400)

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
        if sell_product_dict.has_key(sp):
            return sell_product_dict[sp]
        elif (isinstance(sp, unicode) and sp.encode('utf-8') != '\N'):
            return sell_product_dict[u'其他']
        else:
            return np.nan
    def do_clean_train(row, data_type):
        row = row.values.astype(str)
        for item in row:
            if len(item) == 0 or item == 'nan' or item == '\N':
                return False
        return True

    #只取部分列
    user_info_df = user_info_df[columns_dict.keys()]

    user_info_df = user_info_df.rename(columns=columns_dict)

    user_info_df['from_num'] = user_info_df['from_num'].apply(lambda num: num if len(num)>=30 and len(num)<=32 else np.nan)

    user_info_df['user_type'] = user_info_df['user_type'].apply(lambda type: type_dict[type] if type_dict.has_key(type) else np.nan)

    if(data_type == DataType.NORMAL.value or data_type == DataType.TEST.value):
        user_info_df['open_tate'] = user_info_df['open_tate'].apply(parse_time_normal)
    else:
        user_info_df['open_tate'] = user_info_df['open_tate'].apply(parse_time_fraud)

    user_info_df['sub_stattp'] = user_info_df['sub_stattp'].apply(lambda stat: sub_stat_dict[stat] if sub_stat_dict.has_key(stat) else np.nan)

    user_info_df['is_realname'] = user_info_df['is_realname'].apply(lambda is_real: is_real if str(is_real) == '0' or str(is_real) == '1' or is_real == 0 or is_real == 1 else np.nan)

    user_info_df['sell_product'] = user_info_df['sell_product'].apply(parse_sell_product)

    if(data_type == DataType.TEST.value):
        user_info_df = user_info_df.fillna(0)
        user_info_df['is_include'] = True
    else:
        user_info_df['is_include'] = user_info_df.apply(do_clean_train, axis=1)

    user_info_df_include = user_info_df[user_info_df.is_include == True]
    user_info_df_exclude = user_info_df[user_info_df.is_include == False]

    user_info_df_include = user_info_df_include.drop_duplicates('from_num')

    user_info_dict = trans_dict(user_info_df_include)

    return user_info_df_include, user_info_df_exclude, user_info_dict

def trans_dict(user_info_df_include):
    user_info_dict = {}
    def make_dict(row):
        user_info_dict[row['from_num']] = row.copy()
    user_info_df_include.apply(make_dict, axis=1)
    return user_info_dict


if __name__ == '__main__':
    #filterProcess('/Users/mayuchen/Documents/Python/Repository/DL/UserBehaviorMining/resource/little_data/raw/info/list360诈骗电话-0512_md5.xlsx', DataType.FRAUD.value)
    filterProcess('/Users/mayuchen/Documents/Python/Repository/DL/UserBehaviorMining/resource/little_data/raw/info/号码清单1_md5.xlsx', DataType.TEST.value)
