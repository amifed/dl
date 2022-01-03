import os
import sys
import time
from typing import Callable


def is_idle() -> bool:
    pid = os.popen(
        "ps -ef | grep train.py | grep -v grep | awk  '{print $2}' | xargs").read().replace('\n', '').split(' ')
    pid = list(filter(lambda x: x != '', pid))
    return len(pid) == 0


def run(command: Callable):

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
    command = command(argv_str, result_dir, log_path)
    print(command)
    os.system(command)


# args = ['-n sppb_resnet.resnet18 -e 128 -lr 0.001',
#         '-n hcam_resnet.resnet18 -e 128 -lr 0.001', ]

# args = [
#     '-n hcam_resnet.resnet18 -e 150 -lr 0.001 -sm',
#     '-n hcam_sppb_resnet.resnet18 -e 150 -lr 0.001 -sm',
#     '-n hcam_sppb_resnet_alexnet.resnet18 -e 256 -lr 0.001 -pl -sm',
#     '-n alexnet.alexnet -e 150 -lr 0.001 -sm',
#     '-n resnet.resnet34 -e 150 -lr 0.001 -sm',
#     '-n ca_resnet.resnet18 -e 150 -lr 0.001 -sm',
#     '-n resnet.resnet18 -e 150 -lr 0.001 -sm',
# ]

# 迁移学习
args = [
    '-e 150 -n hcam_sppb_resnet.resnet18 -pt -lr 0.001 -sm -pth /home/djy/dl/result/2021-12-31/08:13:19/model.pth -m continue to train from /home/djy/dl/result/2021-12-31/08:13:19/model.pth'
    '-n sppb_resnet.resnet18 -e 64 -bs 40 -lr 0.001 -pt -sm',
]

commands = list(map(lambda arg: (lambda argv_str, result_dir,
                                 log_path: f'nohup python3 train.py {argv_str} {arg} -p {result_dir} > {log_path} &'), args))


def work():
    if is_idle():
        global idx
        idx += 1
        print(f'run command {idx}')
        run(commands[idx])


if __name__ == '__main__':
    print('\n\n\n')
    global idx
    idx = -1

    while True:
        work()
        time.sleep(10)
