import os
import sys
import time
import getopt

# 记录执行时的当前时间
now = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))
month_day = time.strftime("%Y-%m-%d", time.localtime(time.time()))
time = time.strftime("%H:%M:%S", time.localtime(time.time()))

root = os.path.abspath('.')

month_day_dir = os.path.join(root, 'result', month_day)
if not os.path.exists(month_day_dir):
    os.mkdir(month_day_dir)

result_dir = os.path.join(month_day_dir, time)
log_path = os.path.join(result_dir, 'log.out')

# 新建输出文件夹
os.mkdir(result_dir)

# 调用系统命令，后台训练模型，训练 log 输出到 log_path 中
command = f'nohup python3 train.py -d {result_dir} -S > {log_path} &'
os.system(command)
