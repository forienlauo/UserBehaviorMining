# coding=utf-8
import random


class ExampleGenerator(object):
    def __init__(
            self,
            height=None, weight=None, channels=None,
            exampleCnt=None, positiveExampleProportion=None, trainExampleProportion=None,
            trainDataFilePath=None, testDataFilePath=None,
    ):
        super(ExampleGenerator, self).__init__()
        self.imageConf = ExampleGenerator.ImageConf(height, weight, channels, )
        self.examleConf = ExampleGenerator.ExamleConf(
            exampleCnt, positiveExampleProportion, trainExampleProportion,
            trainDataFilePath, testDataFilePath,
        )

    class ImageConf(object):
        __DEFAULT_HEIGHT = 5 * 60
        __DEFAULT_WEIGHT = 10
        __DEFAULT_CHANNELS = 1

        def __init__(self, height=None, weight=None, channels=None, ):
            super(ExampleGenerator.ImageConf, self).__init__()
            self.height = height or ExampleGenerator.ImageConf.__DEFAULT_HEIGHT
            self.weight = weight or ExampleGenerator.ImageConf.__DEFAULT_WEIGHT
            self.channels = channels or ExampleGenerator.ImageConf.__DEFAULT_CHANNELS

    class ExamleConf(object):
        __DEFAULT_EXAMPLE_CNT = 1000

        __DEFAULT_POSITIVE_EXAMPLE_PROPORTION = 0.1
        __DEFAULT_TRAIN_EXAMPLE_PROPORTION = 0.8

        __DEFAULT_TRAIN_DATA_FILE_PATH = 'tmp/cnn_test/train_data.in'
        __DEFAULT_TEST_DATA_FILE_PATH = 'tmp/cnn_test/test_data.in'

        def __init__(
                self,
                exampleCnt=None, positiveExampleProportion=None, trainExampleProportion=None,
                trainDataFilePath=None, testDataFilePath=None,
        ):
            super(ExampleGenerator.ExamleConf, self).__init__()
            self.exampleCnt = exampleCnt or ExampleGenerator.ExamleConf.__DEFAULT_EXAMPLE_CNT
            self.positiveExampleProportion = positiveExampleProportion or ExampleGenerator.ExamleConf.__DEFAULT_POSITIVE_EXAMPLE_PROPORTION
            self.trainExampleProportion = trainExampleProportion or ExampleGenerator.ExamleConf.__DEFAULT_TRAIN_EXAMPLE_PROPORTION
            self.trainDataFilePath = trainDataFilePath or ExampleGenerator.ExamleConf.__DEFAULT_TRAIN_DATA_FILE_PATH
            self.testDataFilePath = testDataFilePath or ExampleGenerator.ExamleConf.__DEFAULT_TEST_DATA_FILE_PATH

        @staticmethod
        def newProportion():
            return 1.0 * random.randint(0, 99) / 100

    def generate(self, ):
        with open(self.examleConf.trainDataFilePath, 'w') as train_data_file, \
                open(self.examleConf.testDataFilePath, 'w') as test_data_file:
            for _example_iter in range(self.examleConf.exampleCnt):
                cols = []
                for _i in range(self.imageConf.height):
                    for _j in range(self.imageConf.weight):
                        for _k in range(self.imageConf.channels):
                            cols.append(random.randint(1, 100))

                if ExampleGenerator.ExamleConf.newProportion() < self.examleConf.positiveExampleProportion:  # 1%的概率是诈骗
                    cols.extend([1, 0])
                else:  # 其余是非诈骗
                    cols.extend([0, 1])

                line = '\t'.join(map(lambda x: str(x), cols)) + '\n'
                if ExampleGenerator.ExamleConf.newProportion() < self.examleConf.trainExampleProportion:  # 80%的样本作为训练数据
                    train_data_file.write(line)
                else:  # 其余是测试数据
                    test_data_file.write(line)
            return self.get_data_file_paths
        raise IOError('Failed to generate examples using givin conf.')

    def get_data_file_paths(self):
        return (self.examleConf.trainDataFilePath, self.examleConf.testDataFilePath,)


if __name__ == '__main__':
    ExampleGenerator(
        # height=5 * 60,
        # weight=20,
    ).generate()
