import os
import sys
import time
import getopt

# 记录执行时的当前时间
now = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))

root = os.path.abspath('.')
result_dir = os.path.join(root, 'result', now)
log_path = os.path.join(result_dir, 'log.out')
# 新建输出文件夹
os.mkdir(result_dir)

# 调用系统命令，后台训练模型，训练 log 输出到 log_path 中
command = f'nohup python3 train.py -d {result_dir} > {log_path} &'
os.system(command)
