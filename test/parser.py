import argparse


parser = argparse.ArgumentParser()

# message
parser.add_argument('-m', '--msg',
                    dest='msg')
# 并行 CNN 训练
parser.add_argument('-pl', '--parallel',
                    dest='parallel', action='store_true')
# 预训练参数
parser.add_argument('-pt', '--pretained',
                    dest='pretained', action='store_true')
# epoch
parser.add_argument('-e', '--epoch',
                    dest='epoch', default=64, type=int)
# 保存模型路径
parser.add_argument('-sm', '--save-model',
                    dest='save_model')
# 文件保存路径
parser.add_argument('-p', '--path',
                    dest='path')


args = vars(parser.parse_args())

print(args['epoch'])
