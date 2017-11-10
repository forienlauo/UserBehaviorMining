import conf
import os
import logging
from conf import DataType
from src.util.filter.InfoFilter import InfoFilter
from src.util.filter.RecordFilter import RecordFilter
from src.util.feature.FeatureCNN import FeaturesCNN


class Prepare:

    @staticmethod
    def print_usage():
        print("""
 need args:
    <do_clean>  (Empty the temporary data, 1: yes, 0: no)
    <info_data_path>
    <record_data_path>
    <output_data_path>
""")

    @staticmethod
    def prepare(argv):
        if len(argv) < 3:
            Prepare.print_usage()
            return 1
        logging.info('argv: "%s"', ' '.join(argv))

        info_data_path, record_data_path, output_data_path = argv[0], argv[1], argv[2]
        do_clean = argv[3]


        if not os.path.exists(info_data_path):
            logging.info('info_data_path do not exists: "%s"', info_data_path)
            return -1

        if not os.path.exists(record_data_path):
            logging.info('record_data_path do not exists: "%s"', record_data_path)
            return -1

        if not os.path.exists(output_data_path):
            logging.info('output_data_path do not exists: "%s"', output_data_path)
            return -1

        if (do_clean == '1'):
            if (os.path.exists(output_data_path)):
                os.removedirs(output_data_path)
            os.makedirs(output_data_path)


        normal_info = InfoFilter(DataType.normal.name, info_data_path , output_data_path).process()
        normal_record = RecordFilter(DataType.normal.name, record_data_path, output_data_path).process()
        normal_feature = FeaturesCNN(DataType.normal.name, output_data_path)
        normal_feature.set_info_filter(normal_info)
        normal_feature.set_record_filter(normal_record)
        normal_feature.process()

        fraud_info = InfoFilter(DataType.fraud.name, info_data_path, output_data_path).process()
        fraud_record = RecordFilter(DataType.fraud.name, record_data_path , output_data_path).process()
        fraud_feature = FeaturesCNN(DataType.fraud.name, output_data_path)
        fraud_feature.set_info_filter(fraud_info)
        fraud_feature.set_record_filter(fraud_record)
        fraud_feature.process()


if __name__ == '__main__':
    argv = ['1', \
            '/Users/mayuchen/Documents/Python/Repository/DL/Other/UserBehaviorMining/resource/raw/',
            '/Users/mayuchen/Documents/Python/Repository/DL/Other/UserBehaviorMining/resource/raw/',
            '/Users/mayuchen/Documents/Python/Repository/DL/Other/UserBehaviorMining/resource/data/'
            ]
    Prepare.prepare(argv)








