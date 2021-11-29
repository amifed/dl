import torch
from torch.nn import Conv2d, MaxPool2d, Module, Linear
from torch.nn.functional import relu
import torch.nn as nn

__all__ = ['LeNet']


class LeNet(Module):
    # leNet
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # # 定义第一层卷积层
        # self.conv1 = Conv2d(3, 16, 5)
        # # 定义下采样层（池化），改变宽高
        # self.pool1 = MaxPool2d(2, 2)
        # # 定义第二层卷积
        # self.conv2 = Conv2d(16, 32, 5)
        # # 定义下采样层（池化），改变宽高
        # self.pool2 = MaxPool2d(2, 2)
        # # 全连接层
        # self.fc1 = Linear(32 * 5 * 5, 120)
        # self.fc2 = Linear(120, 84)
        # self.fc3 = Linear(84, num_classes)
        self.feature = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        # x = relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        # x = self.pool1(x)            # output(16, 14, 14)
        # x = relu(self.conv2(x))    # output(32, 10, 10)
        # x = self.pool2(x)            # output(32, 5, 5)
        # # 将三维数据 flat
        # x = x.view(-1, 32*5*5)       # output(32*5*5)
        # x = relu(self.fc1(x))      # output(120)
        # x = relu(self.fc2(x))      # output(84)
        # x = self.fc3(x)              # output(10)

        x = self.feature(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
