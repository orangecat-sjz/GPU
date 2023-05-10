#!/bin/bash

#SBATCH -n 12 # 指定核心数量
#SBATCH -N 1 # 指定node的数量
#SBATCH -t 0-5:00 # 运行总时间，天数-小时数-分钟， D-HH:MM
#SBATCH -p debug # 提交到哪一个分区
#SBATCH --gres=gpu:1 # 需要使用多少GPU，n是需要的数量
./graphsampling
