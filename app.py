import os
import sys
import time
import getopt

now = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))

root = os.path.abspath('.')
result_dir = os.path.join(root, 'result', now)
log_path = os.path.join(result_dir, 'log.out')
os.mkdir(result_dir)

command = f'nohup python3 main.py -d {result_dir} > {log_path} &'
os.system(command)
