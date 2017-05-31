#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import datetime

def interval_in_day(time, interval):
    dt = datetime.datetime.strptime(str(time), "%Y%m%d%H%M%S")
    dt_base = datetime.datetime(dt.year, dt.month, dt.day, 0, 0, 0)
    return int((dt - dt_base).total_seconds() / (interval * 60))

