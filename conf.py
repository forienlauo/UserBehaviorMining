# coding=utf-8
import datetime
import json
import logging
import os
import sys
from multiprocessing import cpu_count
from enum import Enum, unique

# log setting
__LOG_LEVEL = (logging.INFO, logging.ERROR, logging.DEBUG)[-1]

__LOG_FORMAT = '%(levelname)5s %(asctime)s [%(filename)s line:%(lineno)d] %(message)s'

logging.basicConfig(format=__LOG_FORMAT, level=__LOG_LEVEL)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

CPU_COUNT = cpu_count()

@unique
class DataType(Enum):
    normal = 0
    fraud = 1
    test = 2


class FeatureCNNConf(object):
    SEPARATOR = '|'

    REPETITION_COUNTS = 7

    FEATURE_COUNTS = 11

    COLUMNS_NAME = [
        'fe_cost_mean',
        'fe_duration_mean',
        'fe_all_call_count_count',
        'fe_type_median',
        'fe_cost_std',
        'fe_duration_std'
    ]


class FeatureSVMConf(object):

    COLUMNS_NAME = [
        'is_realname',
        'sell_product',
        'user_type',
        'open_tate',
        'sub_stattp',
        'plan_name',
        'is_include'
    ]


class RecordConf(object):
    SEPARATOR = '\t'

    INTERVAL = 5   # 5分钟为采样间隔

    FEATURE_NUM = 10
    PARTITION_COUNT = 200

    RECORD_TABLE_IND = dict(
        from_num=[0, int],
        to_num=[1, int],
        charge_num=[2, int],
        start_time=[3, int],
        duration=[4, int],
        cost=[5, int],
        # FIXME 20171213 remove column 'type'
        type=[6, int],
        call_type=[7, int],
        longdis_type=[8, int],
        roaming_type=[9, int],
        voicecall_type=[10, int],
    )

    TYPE_DICT = {
        'cvoi': '0',
        'local': '1',
        'distance': '2'
    }  # type类型映射

    SAMPLING_MONTH = (1, 13)  # 采样月份的范围
    SAMPLING_HOUR = [(7, 12), (12, 17), (17, 22)]  # 每天采样的时间段，单位为小时


class InfoConf(object):

    INFO_TABLE_IND = {
        u'主叫': 'from_num',
        # u'用户名（userName）':'user_name',
        u'产品名（planName）': 'plan_name',
        # u'客户标识（custCode）':'cust_code',
        u'用户类别（userType）': 'user_type',
        u'开户时间（openDate）': 'open_tate',
        u'是否停机（subStatTp）': 'sub_stattp',
        u'是否实名制（isRealname）': 'is_realname',
        u'销售产品（sellProduct）': 'sell_product',
        # u'360标注类型（telType）':'tel_type',
        # u'360标注次数（telCount）':'tel_count',
        # u'业务行为（xsub_action_cd）':'xsub_action_cd',
        # u'停机日期（xstop_dt）':'xstop_dt'
    }

    INFO_TYPE_DICT = {
        u'政企': 0,
        u'公众': 100
    }

    INFO_SUB_STAT_DICT = {
        'BLANK' : 0,
        u'活动': 40,
        u'停机': 80,
        u'拆机': 120,
        u'帐务停机': 160,
        u'割接': 200
    }

    SELL_PRODUCT_DICT = {
        u'CDMA预付费': 0,
        u'CDMA后付费': 30,
        u'CDMA准实时预付费': 60,
        u'C+W（E+W）预付费': 90,
        u'C+W（E+W）后付费': 120,
        u'C+W（E+W）准实时预付费': 150,
        u'其他': 180
    }


class CDRDataDict(object):
    DEFAULT_SEPARATOR = '|'
    DEFAULT_FEATURE_IDX_FILE_PATH = os.path.join(ROOT_DIR, 'resource/original_data/cdr-feature_dict.json')

    # enum
    COMMUNICATION_TYPE_ENUMS = {
        '本地': 0,
        '国内长途': 1,
        '漫游国内': 2,
    }
    CALL_TYPE_ENUMS = {
        '主叫': 0,
        '被叫': 1,
        '呼转': 2,
    }
    __feature_idxs = dict()

    @staticmethod
    def init(default_feature_idx_file_path=None):
        if default_feature_idx_file_path:
            with open(default_feature_idx_file_path) as default_feature_idx_file:
                default_feature_idx = json.loads(default_feature_idx_file.read())
                CDRDataDict.reset_feature_idxs_by_user(default_feature_idx)

    @staticmethod
    def get_feature_num():
        return len(CDRDataDict.__feature_idxs)

    @staticmethod
    def add_feature(feature_name):
        if feature_name in CDRDataDict.__feature_idxs:
            raise Exception("Duplicated feature_name: %s." % feature_name)
        CDRDataDict.__feature_idxs[feature_name] = len(CDRDataDict.__feature_idxs)
        return CDRDataDict

    @staticmethod
    def remove_feature(feature_name, ignore_whether_exist=True):
        if not ignore_whether_exist:
            if feature_name not in CDRDataDict.__feature_idxs:
                raise Exception("Unknown feature_name: %s." % feature_name)
        poped_idx = CDRDataDict.__feature_idxs.pop(feature_name, sys.maxint)
        # correct
        for tmp_feature_name in CDRDataDict.__feature_idxs:
            if CDRDataDict.__feature_idxs[tmp_feature_name] > poped_idx:
                CDRDataDict.__feature_idxs[tmp_feature_name] -= 1
        return CDRDataDict

    @staticmethod
    def reset_feature_idxs_by_user(feature_idxs_or_feature_names):
        if isinstance(feature_idxs_or_feature_names, (list, tuple)):
            feature_names = feature_idxs_or_feature_names
            for feature_name in feature_names:
                CDRDataDict.add_feature(feature_name)
        elif isinstance(feature_idxs_or_feature_names, dict):
            feature_idxs = feature_idxs_or_feature_names
            CDRDataDict.__feature_idxs = dict(feature_idxs)
        else:
            raise TypeError("Cannot reset feature_idxs with type: %s." % type(feature_idxs_or_feature_names))
        return CDRDataDict.__feature_idxs

    @staticmethod
    def get_copy_of_feature_idxs():
        return dict(CDRDataDict.__feature_idxs)

    @staticmethod
    def get_feature_idx(feature_name):
        feature_name = str(feature_name)
        if feature_name not in CDRDataDict.__feature_idxs:
            raise StandardError("Unknown feature_name: %s." % feature_name)
        return CDRDataDict.__feature_idxs[feature_name]

    @staticmethod
    def get_feature_name(feature_idx):
        if feature_idx < 0 or feature_idx >= len(CDRDataDict.__feature_idxs):
            raise StandardError("Illegal feature_idx(must ge %d and lt %d): %d."
                                % (0, len(CDRDataDict.__feature_idxs), feature_idx))
        for feature_name in CDRDataDict.__feature_idxs.keys():
            tmp_feature_idx = CDRDataDict.__feature_idxs[feature_name]
            if tmp_feature_idx == feature_idx:
                return feature_name
        raise StandardError("Unknown error.")

    @staticmethod
    def get_ordered_feature_names(reverse=False):
        items = CDRDataDict.__feature_idxs.items()
        items.sort(lambda item1, item2: cmp(int(item1[1]), int(item2[1])), reverse=reverse)
        ordered_feature_names = [item[0] for item in items]
        return ordered_feature_names


class ProvinceCityLocationDict(object):
    __LOCATION_FILE_PATH = os.path.join(ROOT_DIR, 'configuration/中国主要城市经纬度查询(精确到二级省市).txt')
    __pos2location_dict = None

    @staticmethod
    def get_locations_by_pos(pos):
        # lazy load
        if ProvinceCityLocationDict.__pos2location_dict is None:
            ProvinceCityLocationDict.init()
        return ProvinceCityLocationDict.__pos2location_dict[pos]

    @staticmethod
    def init():
        useless_set = {'台湾省', }
        tequ_set = {'香港特别行政区', '澳门特别行政区', }
        zhixianshi_set = {'北京市', '上海市', '天津市', '重庆市', }
        zizhiqu_set = {'广西自治区', '内蒙古自治区', '宁夏自治区', '西藏自治区', '新疆自治区', }
        with open(ProvinceCityLocationDict.__LOCATION_FILE_PATH) as location_file:
            ProvinceCityLocationDict.__pos2location_dict = dict()
            for line in location_file:
                elems = [elem.strip() for elem in line.strip().split(' ')]
                province = elems[0]
                city = elems[1]
                lat = elems[2]
                lng = elems[3]

                # pos
                if province in useless_set:
                    continue
                elif province in tequ_set:
                    province = province.split('特别行政区')[0]
                    pos = province
                elif province in zhixianshi_set:
                    province = province.split('市')[0]
                    if province == '重庆' and city != '重庆市':
                        continue
                    pos = province
                elif province in zizhiqu_set:
                    province = province.split('自治区')[0]
                    pos = '.'.join([province, city])
                else:  # normal province name
                    province = province.split('省')[0]
                    pos = '.'.join([province, city])

                # location
                if lat.startswith('北纬'):
                    lat = float(lat.split('纬')[1])
                elif lat.startswith('南纬'):
                    lat = float('-' + lat.split('纬')[1])
                if lng.startswith('东经'):
                    lng = float(lng.split('经')[1])
                elif lng.startswith('西经'):
                    lng = float('-' + lng.split('经')[1])

                if pos in ProvinceCityLocationDict.__pos2location_dict:
                    raise Exception("Duplicate pos: %s" % pos)
                ProvinceCityLocationDict.__pos2location_dict[pos] = (lng, lat)


class ProvinceAvgIncomeDict(object):
    __AVG_INCOME_FILE_PATH = os.path.join(ROOT_DIR, 'configuration/province2avg_income.txt')
    __province2avg_income_dict = None

    @staticmethod
    def get_avg_income_by_pos(pos):
        # lazy load
        if ProvinceAvgIncomeDict.__province2avg_income_dict is None:
            ProvinceAvgIncomeDict.init()
        return ProvinceAvgIncomeDict.__province2avg_income_dict[pos]

    @staticmethod
    def init():
        with open(ProvinceAvgIncomeDict.__AVG_INCOME_FILE_PATH) as avg_income_file:
            ProvinceAvgIncomeDict.__province2avg_income_dict = dict()
            for line in avg_income_file:
                elems = [elem.strip() for elem in line.strip().split(' ')]
                province = elems[0]
                avg_income = elems[1]
                ProvinceAvgIncomeDict.__province2avg_income_dict[province] = float(avg_income)


def init():
    CDRDataDict.init()
    ProvinceCityLocationDict.init()
    ProvinceAvgIncomeDict.init()


if __name__ == '__main__':
    # logging.error(ProvinceCityLocationDict.get_locations_by_pos('安徽.宿州'))
    # logging.error(ProvinceCityLocationDict.get_locations_by_pos('广东.深圳'))
    # logging.error(ProvinceCityLocationDict.get_locations_by_pos('上海'))
    # logging.error(ProvinceCityLocationDict.get_locations_by_pos('内蒙古.呼和浩特'))
    # logging.error(ProvinceCityLocationDict.get_locations_by_pos('江苏.徐州'))
    # logging.error(ProvinceCityLocationDict.get_locations_by_pos('湖北.石堰'))
    logging.error(ProvinceAvgIncomeDict.get_avg_income_by_pos('湖北'))
    logging.error(ProvinceAvgIncomeDict.get_avg_income_by_pos('北京'))
