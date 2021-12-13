from typing import Callable
import torch
from torch.functional import Tensor
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.optim as optim
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
import os
import argparse
import time
from utils.device import device
from utils.data import ParallelImageFolder

transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=(0, 180)),
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
        save_model: bool,
        path: str,
        **kwargs):

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
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    print(f'loss_function = {criterion}\noptimizer = {optimizer}\n')

    # 4. train
    max_accuracy, max_accuracy_epoch = 0.0, 0
    min_loss, min_loss_epoch = float('inf'), 0

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

            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader)
        print('Loss: %.3f' % (epoch_loss), end='\t')

        if epoch_loss < min_loss:
            min_loss, min_loss_epoch = epoch_loss, epoch

        """valid"""
        net.eval()
        correct = 0
        total = 0
        # count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

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

                for label, prediction in zip(labels, predictions):
                    if prediction == label:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        epoch_acc = 100 * correct / total
        print('Accuracy: %.2f%%' % (epoch_acc), end='\t')
        if epoch_acc > max_accuracy:
            max_accuracy, max_accuracy_epoch = epoch_acc, epoch
            save_path = os.path.join(path, 'model.pth')
            if save_model:
                torch.save(net.state_dict(), save_path)
        print('Cost %ds' % (time.time() - now))

        # for classname, correct_cnt in correct_pred.items():
        #     accuracy = 100 * float(correct_cnt) / total_pred[classname]
        #     print('Accuracy for class {:5s} is: {:.1f} %'.format(
        #         classname, accuracy), end='\n')

    print('\nFinished training!!!', end='\n\n')
    print('Min Loss = %.3f in epoch %d;\n\
    Max Accuracy = %.2f%% in epoch %d;\n\
    Total Cost %d minutes' % (min_loss,
                              min_loss_epoch,
                              max_accuracy,
                              max_accuracy_epoch,
                              (time.time() - start) / 60), end='\n\n')

    print(net, end='\n\n')


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
    pretrained, parallel, msg = args['pretrained'], args['parallel'], args['msg']
    dataset_path = '/home/djy/dataset/dataset'
    seg_dataset_path = '/home/djy/dataset/ycrcb_hsv_dataset'
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
        dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(2))

    train(trainset, validset, resnet.resnet18, **args)
