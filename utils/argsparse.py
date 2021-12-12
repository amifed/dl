import argparse


def args():
    parser = argparse.ArgumentParser()

    # 并行 CNN 训练
    parser.add_argument('-pl', '--parallel',
                        dest='parallel', action='store_true')
    # 预训练参数
    parser.add_argument('-pt', '--pretained',
                        dest='pretained', action='store_true')
    # epoch
    parser.add_argument('-e', '--epoch',
                        dest='epoch', default=64, type=int)
    # batch size
    parser.add_argument('-bs', '--batch-size',
                        dest='batch_size', default=20, type=int)
    # 保存模型路径
    parser.add_argument('-sm', '--save-model',
                        dest='save_model', action='store_true')
    # 文件保存路径
    parser.add_argument('-p', '--path',
                        dest='path')

    # message
    parser.add_argument('-m', '--msg',
                        dest='msg', nargs='*')

    return vars(parser.parse_args())
