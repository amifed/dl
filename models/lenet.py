import torch
import torch.nn as nn
from torch.nn.functional import relu


class LeNet(nn.Module):
    # leNet
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
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
