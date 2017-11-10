#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import os


class BaseProcess:

    def __init__(self, user_type, output_data_path):
        self.user_type = user_type
        self.output_data_path = output_data_path

    def mkdirs(self, path):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def get_output_path(self, basename = '', clude_type = ''):
        '''
        :param content_type: info/record
        :param clude_type: 过滤可用数据/过滤布可用数据
        :return:
        '''
        return os.path.join(self.output_data_path, self.process_name, self.content_type, clude_type, basename)

    def process(self):
        raise NotImplementedError, "Class %s does not implement process(self)" % self.__class__
        pass

    def check_data(self):
        raise NotImplementedError, "Class %s does not implement check_data(self)" % self.__class__
        pass

