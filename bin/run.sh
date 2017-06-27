#!/usr/bin/env bash

script_name=$0
dir=`dirname $script_name`
cd $dir/..

log_file="log/`date +%Y-%m-%d-%H-%M-%S`.log"

nohup python start.py $@ >> $log_file 2>&1 &
