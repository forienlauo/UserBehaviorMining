import os

import math
from util import common


def split_by_proportion(src_file, target_dir, split_file_cnt):
    factor = len(str(int(split_file_cnt))) * 10
    with open(src_file) as file:
        split_files = [open(os.path.join(target_dir, 'part-%d.split' % file_no), 'w')
                       for file_no in range(split_file_cnt)]
        for line in file:
            file_no = int(math.floor(common.new_proportion() * factor)) % split_file_cnt
            split_files[file_no].write(line)

        [split_file.close() for split_file in split_files]
