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
    __DEFAULT_EXAMPLE_CNT = 1000
    __DEFAULT_POSITIVE_EXAMPLE_PROPORTION = 0.1

    __DEFAULT_GENERATED_DATA_FILE_PATH = os.path.join(conf.ROOT_DIR, 'tmp/cnn_test/data.in')

    def __init__(
            self,
            height=None, weight=None, channels=None,
            exampleCnt=None, positiveExampleProportion=None, dataFilePath=None,
    ):
        super(ExampleGenerator, self).__init__()
        self.imageConf = ImageConf(height, weight, channels, )
        self.exampleCnt = exampleCnt or ExampleGenerator.__DEFAULT_EXAMPLE_CNT
        self.positiveExampleProportion = positiveExampleProportion or ExampleGenerator.__DEFAULT_POSITIVE_EXAMPLE_PROPORTION
        self.generatedDataFilePath = dataFilePath or ExampleGenerator.__DEFAULT_GENERATED_DATA_FILE_PATH

    def generate(self, ):
        with open(self.generatedDataFilePath, 'w') as generatedDataFile:
            for _example_iter in range(self.exampleCnt):
                cols = []
                for _i in range(self.imageConf.height):
                    for _j in range(self.imageConf.weight):
                        for _k in range(self.imageConf.channels):
                            cols.append(random.randint(1, 100))

                if common.new_proportion() < self.positiveExampleProportion:  # 1%的概率是诈骗
                    cols.extend([1, 0])
                else:  # 其余是非诈骗
                    cols.extend([0, 1])

                line = '\t'.join(map(lambda x: str(x), cols)) + '\n'
                generatedDataFile.write(line)
            return self.generatedDataFilePath
        raise IOError('Failed to generate examples using givin conf.')


class ExampleAllocator(object):
    __DEFAULT_TRAIN_EXAMPLE_PROPORTION = 0.8

    __DEFAULT_ALLOCATED_TRAIN_DATA_FILE_PATH = os.path.join(conf.ROOT_DIR, 'tmp/cnn_test/train_data.in')
    __DEFAULT_ALLOCATED_TEST_DATA_FILE_PATH = os.path.join(conf.ROOT_DIR, 'tmp/cnn_test/test_data.in')

    def __init__(
            self,
            wholeDataFilePath,
            trainExampleProportion=None, trainDataFilePath=None, testDataFilePath=None,
    ):
        super(ExampleAllocator, self).__init__()
        self.wholeDataFilePath = wholeDataFilePath
        self.trainExampleProportion = trainExampleProportion or ExampleAllocator.__DEFAULT_TRAIN_EXAMPLE_PROPORTION
        self.allocatedTrainDataFilePath = trainDataFilePath or ExampleAllocator.__DEFAULT_ALLOCATED_TRAIN_DATA_FILE_PATH
        self.allocatedTestDataFilePath = testDataFilePath or ExampleAllocator.__DEFAULT_ALLOCATED_TEST_DATA_FILE_PATH

    def allocate(self, ):
        with open(self.wholeDataFilePath) as whole_data_file, \
                open(self.allocatedTrainDataFilePath, 'w') as allocatedTrainDataFile, \
                open(self.allocatedTestDataFilePath, 'w') as allocatedTestDataFile:
            for line in whole_data_file:
                if common.new_proportion() < self.trainExampleProportion:  # 80%的样本作为训练数据
                    allocatedTrainDataFile.write(line)
                else:  # 其余是测试数据
                    allocatedTestDataFile.write(line)
            return self.allocatedTrainDataFilePath, self.allocatedTestDataFilePath
        raise IOError('Failed to allocate examples using givin conf.')


if __name__ == '__main__':
    conf.init()
    # generatedDataFilePath = ExampleGenerator(
    #     height=5 * 60,
    #     weight=20,
    # ).generate()
    generatedDataFilePath = '/Users/lauo/PycharmProjects/LinkLand/UserBehaviorMining/tmp/cnn_test/data.in'
    allocatedTrainDataFilePath, allocatedTestDataFilePath = ExampleAllocator(
        generatedDataFilePath,
    ).allocate()
