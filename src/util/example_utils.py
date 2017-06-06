# coding=utf-8
import os
import random

import conf
from src.util import common


class ImageConf(object):
    __DEFAULT_HEIGHT = 5 * 60
    __DEFAULT_WEIGHT = 10
    __DEFAULT_CHANNELS = 1

    def __init__(self, height=None, weight=None, channels=None, ):
        super(ImageConf, self).__init__()
        self.height = height or ImageConf.__DEFAULT_HEIGHT
        self.weight = weight or ImageConf.__DEFAULT_WEIGHT
        self.channels = channels or ImageConf.__DEFAULT_CHANNELS


class ExampleGenerator(object):
    __COL_DELEMITER = '\t'
    __LINE_DELEMITER = '\n'

    __DEFAULT_EXAMPLE_CNT = 1000
    __DEFAULT_POSITIVE_EXAMPLE_PROPORTION = 0.1

    __DEFAULT_GENERATED_DATA_FILE_PATHS = (
        os.path.join(conf.ROOT_DIR, 'tmp/cnn_test/data_x.in'),
        os.path.join(conf.ROOT_DIR, 'tmp/cnn_test/data_y.in'),
    )

    def __init__(
            self,
            height=None, weight=None, channels=None,
            exampleCnt=None, positiveExampleProportion=None, dataFilePath=None,
    ):
        super(ExampleGenerator, self).__init__()
        self.imageConf = ImageConf(height, weight, channels, )
        self.exampleCnt = exampleCnt or ExampleGenerator.__DEFAULT_EXAMPLE_CNT
        self.positiveExampleProportion = positiveExampleProportion or ExampleGenerator.__DEFAULT_POSITIVE_EXAMPLE_PROPORTION
        self.generatedDataFilePaths = dataFilePath or ExampleGenerator.__DEFAULT_GENERATED_DATA_FILE_PATHS

    def generate(self, ):
        with open(self.generatedDataFilePaths[0], 'w') as generatedDataFile_x, \
                open(self.generatedDataFilePaths[1], 'w') as generatedDataFile_y:
            for _example_iter in range(self.exampleCnt):
                cols_x = []
                for _i in range(self.imageConf.height):
                    for _j in range(self.imageConf.weight):
                        for _k in range(self.imageConf.channels):
                            cols_x.append(random.randint(1, 100))
                line_x = ExampleGenerator.__COL_DELEMITER.join(map(str, cols_x)) + ExampleGenerator.__LINE_DELEMITER
                generatedDataFile_x.write(line_x)

                cols_y = []
                if common.new_proportion() < self.positiveExampleProportion:  # 1%的概率是诈骗
                    cols_y.extend([1, 0])
                else:  # 其余是非诈骗
                    cols_y.extend([0, 1])
                line_y = ExampleGenerator.__COL_DELEMITER.join(map(str, cols_y)) + ExampleGenerator.__LINE_DELEMITER
                generatedDataFile_y.write(line_y)
            return self.generatedDataFilePaths
        raise IOError('Failed to generate examples using givin conf.')


class ExampleAllocator(object):
    __DEFAULT_TRAIN_EXAMPLE_PROPORTION = 0.8

    __DEFAULT_ALLOCATED_TRAIN_DATA_FILE_PATHS = (
        os.path.join(conf.ROOT_DIR, 'tmp/cnn_test/train_data_x.in'),
        os.path.join(conf.ROOT_DIR, 'tmp/cnn_test/train_data_y.in'),
    )
    __DEFAULT_ALLOCATED_TEST_DATA_FILE_PATHS = (
        os.path.join(conf.ROOT_DIR, 'tmp/cnn_test/test_data_x.in'),
        os.path.join(conf.ROOT_DIR, 'tmp/cnn_test/test_data_y.in'),
    )

    def __init__(
            self,
            wholeDataFilePaths,
            trainExampleProportion=None, trainDataFilePaths=None, testDataFilePaths=None,
    ):
        super(ExampleAllocator, self).__init__()
        self.wholeDataFilePaths = wholeDataFilePaths
        self.trainExampleProportion = trainExampleProportion or ExampleAllocator.__DEFAULT_TRAIN_EXAMPLE_PROPORTION
        self.allocatedTrainDataFilePaths = trainDataFilePaths or ExampleAllocator.__DEFAULT_ALLOCATED_TRAIN_DATA_FILE_PATHS
        self.allocatedTestDataFilePaths = testDataFilePaths or ExampleAllocator.__DEFAULT_ALLOCATED_TEST_DATA_FILE_PATHS

    def allocate(self, ):
        with open(self.wholeDataFilePaths[0]) as wholeDataFile_x, \
                open(self.wholeDataFilePaths[1]) as wholeDataFile_y, \
                open(self.allocatedTrainDataFilePaths[0], 'w') as allocatedTrainDataFile_x, \
                open(self.allocatedTrainDataFilePaths[1], 'w') as allocatedTrainDataFile_y, \
                open(self.allocatedTestDataFilePaths[0], 'w') as allocatedTestDataFile_x, \
                open(self.allocatedTestDataFilePaths[1], 'w') as allocatedTestDataFile_y:
            for line_x, line_y in zip(wholeDataFile_x, wholeDataFile_y):
                if common.new_proportion() < self.trainExampleProportion:  # 80%的样本作为训练数据
                    allocatedTrainDataFile_x.write(line_x)
                    allocatedTrainDataFile_y.write(line_y)
                else:  # 其余是测试数据
                    allocatedTestDataFile_x.write(line_x)
                    allocatedTestDataFile_y.write(line_y)
            return self.allocatedTrainDataFilePaths, self.allocatedTestDataFilePaths
        raise IOError('Failed to allocate examples using givin conf.')


if __name__ == '__main__':
    # conf.init()
    # generatedDataFilePaths = ExampleGenerator(
    #     height=5 * 60, weight=20,
    #     exampleCnt=1000,
    # ).generate()
    generatedDataFilePaths = (
        '/Users/lauo/PycharmProjects/LinkLand/UserBehaviorMining/tmp/cnn_test/data_x.in',
        '/Users/lauo/PycharmProjects/LinkLand/UserBehaviorMining/tmp/cnn_test/data_y.in',
    )
    allocatedTrainDataFilePaths, allocatedTestDataFilePaths = ExampleAllocator(
        generatedDataFilePaths,
    ).allocate()
