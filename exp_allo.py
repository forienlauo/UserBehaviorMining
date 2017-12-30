#!/usr/bin/env python
import sys

from src.util.example_utils import ExampleAllocator


def print_usage():
    print("""
need args:
    <data_x_path> <data_y_path>
    <train_data_x_path> <train_data_y_path>
    <test_data_x_path> <test_data_y_path>
    <train_example_proportion>
""")


if __name__ == '__main__':
    if len(sys.argv) <= 7:
        print_usage()
        sys.exit(1)

    data_x_path, data_y_path, \
    train_data_x_path, train_data_y_path, \
    test_data_x_path, test_data_y_path, \
    train_example_proportion \
        = sys.argv[1:8]

    train_example_proportion = float(train_example_proportion)

    ExampleAllocator(
        (data_x_path, data_y_path,),
        trainDataFilePaths=(train_data_x_path, train_data_y_path,),
        testDataFilePaths=(test_data_x_path, test_data_y_path,),
        trainExampleProportion=train_example_proportion,
    ).allocate()

    sys.exit(0)
