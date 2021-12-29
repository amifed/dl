import os
import sys
import time


def run():

    # 记录执行时的当前时间
    now = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))
    month_day = time.strftime("%Y-%m-%d", time.localtime(time.time()))
    timee = time.strftime("%H:%M:%S", time.localtime(time.time()))

    root = os.path.abspath('.')

    month_day_dir = os.path.join(root, 'result', month_day)
    if not os.path.exists(month_day_dir):
        os.mkdir(month_day_dir)

    result_dir = os.path.join(month_day_dir, timee)
    log_path = os.path.join(result_dir, 'log.out')

    # 新建输出文件夹
    os.mkdir(result_dir)

    # 获取命令行参数
    argv_str = ' '.join(sys.argv[1:])

    # 调用系统命令，后台训练模型，训练 log 输出到 log_path 中
    command = f'nohup python3 train.py {argv_str} -p {result_dir} > {log_path} &'
    print(command)
    os.system(command)


if __name__ == '__main__':
    run()
