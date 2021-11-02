import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    # leNet
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # 定义第一层卷积层
        self.conv1 = nn.Conv2d(3, 16, 5)
        # 定义下采样层（池化），改变宽高
        self.pool1 = nn.MaxPool2d(2, 2)
        # 定义第二层卷积
        self.conv2 = nn.Conv2d(16, 32, 5)
        # 定义下采样层（池化），改变宽高
        self.pool2 = nn.MaxPool2d(2, 2)
        # 全连接层
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        # 将三维数据 flat
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x
