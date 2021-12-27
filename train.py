from typing import Callable, List
import torch
from torch.functional import Tensor
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary
from torchstat import stat
from models.lenet import LeNet
from models._alexnet import _alexnet
from models.alexnet import alexnet
import models.resnet as resnet
import models.res_alex_net as res_alex_net
import models.vgg as vgg
import models.densenet as densenet
import models.mobilenet as mobilenet
import models.ghostnet as ghostnet
import models.shufflenetv2 as shufflenetv2
import models.ecanet as ecanet
import models.eca_se_resnet as eca_se_resnet
import models.spp_resnet as spp_resnet
import models.sppb_resnet as sppb_resnet
import models.cbam_resnet as cbam_resnet
import models.cbam_spp_resnet as cbam_spp_resnet
import models.ca_resnet as ca_resnet
import models.ca_spp_resnet as ca_spp_resnet
import models.ca_cbam_resnet as ca_cbam_resnet
import models.cbam_spp_resnet_alexnet as cbam_spp_resnet_alexnet
import models.eca_cbam_resnet as eca_cbam_resnet
import models.ca_cbam_spp_resnet_alexnet as ca_cbam_spp_resnet_alexnet
import models.cbam_resnet_alexnet as cbam_resnet_alexnet
import models.ccam_spp_resnet_alexnet as ccam_spp_resnet_alexnet
import os
import argparse
import time
from sklearn import metrics
from utils.device import device
from utils.data import ParallelImageFolder

basewidth = 320
# wpercent = (basewidth/float(img.size[0]))
# hsize = int((float(img.size[1])*float(wpercent)))
# img = img.resize((320, hsize), Image.ANTIALIAS)

transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=(0, 180)),
    # transforms.Grayscale(),
    # transforms.RandomChoice(
    #     transforms=(
    #         transforms.ColorJitter(brightness=.5, hue=.3),
    #         transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    #         transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    #         transforms.RandomRotation(degrees=(0, 180)),
    #         transforms.RandomPosterize(bits=2)),
    #     p=[0.1, 0.2, 0.1, 0.5, 0.1]
    # ),
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def train(
        trainset: Dataset,
        validset: Dataset,
        network: Callable[..., nn.Module],
        parallel: bool,
        pretrained: bool,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        save_model: bool,
        path: str,
        **kwargs) -> List[float]:

    # 1. dataset load
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=8)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                              shuffle=False, num_workers=8)
    classes = ('bzx', 'cwx', 'hdx', 'mtx', 'nqx', 'qtx', 'zxx')

    # 2. define a CNN
    net: nn.Module = network(num_classes=len(classes), pretrained=pretrained)
    net.to(device)
    print(f'using model: {net.__class__.__name__}, {network.__name__}')
    print(f'using device {device}')
    print(f'batch_size = {batch_size}')
    print(f'epochs = {epochs}')

    # 3. define a loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60], gamma=0.1)

    print(f'loss_function = {criterion}\noptimizer = {optimizer}\n')

    # 4. train
    max_accuracy, max_accuracy_epoch = 0.0, 0
    min_loss, min_loss_epoch = float('inf'), 0
    train_loss_list = []
    valid_accuracy_list = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        now = time.time()

        print(f'\n========== Train Epoch {epoch + 1} ==========', end='\n')

        """train"""
        net.train()
        running_loss = 0.0
        for _, [inputs, labels, parallel_inputs] in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)
            if parallel and parallel_inputs is not None:
                parallel_inputs = parallel_inputs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(
                inputs, parallel_inputs) if parallel else net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader)
        train_loss_list.append(epoch_loss)
        print('Loss: %.3f' % (epoch_loss), end='\t')

        if epoch_loss < min_loss:
            min_loss, min_loss_epoch = epoch_loss, epoch

        """valid"""
        net.eval()
        correct = 0
        total = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels, parallel_images in validloader:

                images, labels = images.to(device), labels.to(device)

                if parallel and parallel_images is not None:
                    parallel_images = parallel_images.to(device)

                outputs: Tensor = net(
                    images, parallel_images) if parallel else net(images)

                _, predictions = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

                y_true += labels.tolist()
                y_pred += predictions.tolist()

        epoch_acc = 100 * correct / total
        valid_accuracy_list.append(100 * correct / total)
        print('Accuracy: %.2f%%' % (epoch_acc), end='\t')
        if epoch_acc > max_accuracy:
            max_accuracy, max_accuracy_epoch = epoch_acc, epoch
            save_path = os.path.join(path, 'model.pth')
            if save_model:
                torch.save(net.state_dict(), save_path)
        print('Cost %ds' % (time.time() - now))

        print(metrics.classification_report(
            y_true, y_pred, digits=4, target_names=classes))
        print(
            f'micro f-score: {metrics.f1_score(y_true, y_pred, average="micro")}')

    print('\nFinished training!!!', end='\n\n')
    print('Min Loss = %.3f in epoch %d;\nMax Accuracy = %.2f%% in epoch %d;\nTotal Cost %d minutes' %
          (min_loss,
           min_loss_epoch,
           max_accuracy,
           max_accuracy_epoch,
           (time.time() - start) / 60), end='\n\n')

    # print(net, end='\n\n')
    summary(net, (3, 320, 320))
    print('\n\n')
    return train_loss_list, valid_accuracy_list


class Plot:
    def __init__(self, label: str, color: str, data: List[float]) -> None:
        self.label = label
        self.color = color
        self.data = data


def plot(epochs: int, title: str, path: str, *args: List[Plot]):
    import matplotlib.pyplot as plt
    plt.clf()
    x = range(epochs)  # epoch
    plt.title(title)
    for plot in args:
        plt.plot(x, plot.data, color=plot.color, label=plot.label)
    plt.xlabel("epoch")  # 横坐标名字
    plt.ylabel("accuracy")  # 纵坐标名字
    plt.legend()
    plt.grid()
    plt.savefig(path)


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    # 并行 CNN 训练
    parser.add_argument('-pl', '--parallel',
                        dest='parallel', action='store_true')
    # 预训练参数
    parser.add_argument('-pt', '--pretrained',
                        dest='pretrained', action='store_true')
    # epoch
    parser.add_argument('-e', '--epochs',
                        dest='epochs', default=64, type=int)
    # batch size
    parser.add_argument('-bs', '--batch-size',
                        dest='batch_size', default=20, type=int)
    # learning rate
    parser.add_argument('-lr', '--learning-rate',
                        dest='learning_rate', default=0.0001, type=float)
    # 保存模型路径
    parser.add_argument('-sm', '--save-model',
                        dest='save_model', action='store_true')
    # 文件保存路径
    parser.add_argument('-p', '--path',
                        dest='path')
    # message
    parser.add_argument('-m', '--msg',
                        dest='msg', nargs='*', type=str)

    args = vars(parser.parse_args())
    pretrained, parallel, msg = args['pretrained'], args['parallel'], args['msg'] or [
    ]
    dataset_path = '/home/djy/dataset/dataset2'
    seg_dataset_path = '/home/djy/dataset/ycrcb_hsv_dataset2'
    print(f'dataset_path: {dataset_path}')
    print(f"pretrained : {pretrained} \nparallel: {parallel}\n")
    if parallel:
        print(f'parallel segmentent dataset : {seg_dataset_path}')
    print(f'msg: {" ".join(msg)}')
    dataset = ParallelImageFolder(
        root=dataset_path, parallel_root=seg_dataset_path, transform=transform)
    full_size = len(dataset)
    train_size = int(0.8 * full_size)
    valid_size = full_size - train_size
    trainset, validset = torch.utils.data.random_split(
        dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(1))

    loss_list, acc_list = train(trainset, validset,
                                ca_resnet.resnet18, **args)

    plot(args['epochs'], "loss for "+" ".join(msg), os.path.join(
        args['path'], 'loss.png'), Plot('loss', 'b', loss_list))
    plot(args['epochs'], "accuracy for "+" ".join(msg), os.path.join(
        args['path'], 'accuracy.png'), Plot('accuracy', 'r', acc_list))
