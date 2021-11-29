from functools import total_ordering
from re import T
from PIL.Image import SAVE, preinit
from matplotlib import image
from matplotlib.pyplot import cla
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from models.lenet import LeNet
from models.alexnet import AlexNet
import models.resnet as resnet
import models.vgg as vgg
import models.densenet as densenet
import models.mobilenet as mobilenet
import models.ghostnet as ghostnet
import models.shufflenetv2 as shufflenetv2
import os
import sys
import time
import getopt
from utils.device import device
from utils.data import ParallelImageFolder

start = time.time()

# save model path from shell args
opts, _ = getopt.getopt(sys.argv[1:], "d:S")
opt_dict = {k: v for [k, v] in opts}
SAVE_MODEL = '-S' in opt_dict
SAVE_PATH = os.path.join(opt_dict.get('-d', './'), 'model.pth')

# 1. load & normalize
transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=(0, 180)),
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

batch_size = 20
# trainset_path = '/home/djy/dataset/dataset'
# testset_path = '/home/djy/dataset/dataset'

# 训练图片
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=False, transform=transform)
# trainset = torchvision.datasets.ImageFolder(
# root = trainset_path, transform = transform)
# 测试图片
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=False, transform=transform)
# testset = torchvision.datasets.ImageFolder(
# root=testset_path, transform=transform)


dataset_path = '/home/djy/dataset/uni_dataset'
seg_dataset_path = '/home/djy/dataset/seg_deeplabv3_dataset'
dataset = ParallelImageFolder(
    root=dataset_path, parallel_root=seg_dataset_path, transform=transform)
full_size = len(dataset)
train_size = int(0.8 * full_size)
test_size = full_size - train_size
trainset, testset = torch.utils.data.random_split(
    dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=8)
# test_dataiter = iter(testloader)
# test_image, test_label = test_dataiter.next()

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# classes = ('bzx', 'cwx', 'hdx', 'mtx', 'nqx', 'nzx', 'qtx', 'sjx', 'zxx')
classes = ('bzx', 'cwx', 'hdx', 'mtx', 'nqx', 'qtx', 'zxx')

# 2. define a CNN
net = AlexNet(num_classes=len(classes))
net.to(device)
print(f'Train Model: {net.__class__.__name__}, Using device {device}')

# 3. define a loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# 4. train
epochs = 64
parallel_train = True
max_accuracy, max_accuracy_epoch = 0.0, 0
min_loss, min_loss_epoch = float('inf'), 0

for epoch in range(epochs):  # loop over the dataset multiple times
    print(f'\n========== Train Epoch {epoch + 1} ==========', end='\n')

    """train"""
    net.train()
    now = time.time()
    running_loss = 0.0
    for i, [inputs, labels, parallel_inputs] in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = inputs.to(device), labels.to(device)
        if parallel_train and parallel_inputs is not None:
            parallel_inputs = parallel_inputs.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(
            inputs, parallel_inputs) if parallel_train else net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
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
        for images, labels, parallel_images in testloader:

            images, labels = images.to(device), labels.to(device)

            if parallel_train and parallel_images is not None:
                parallel_images = parallel_images.to(device)

            outputs = net(images)

            _, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            for label, prediction in zip(labels, predictions):
                if prediction == label:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    epoch_acc = 100 * correct / total
    print('Accuracy: %d %%' % (epoch_acc), end='\t')
    print('Time %d s' % (time.time() - now))
    if (epoch_acc) > max_accuracy:
        max_accuracy, max_accuracy_epoch = epoch_acc, epoch
        if SAVE_MODEL:
            torch.save(net.state_dict(), SAVE_PATH)

    for classname, correct_cnt in correct_pred.items():
        accuracy = 100 * float(correct_cnt) / total_pred[classname]
        print('Accuracy for class {:5s} is: {:.1f} %'.format(
            classname, accuracy), end='\n')

print('Finished training!!!', end='\n\n')
print('Min loss = %.3f in epoch %d;\n\
max Accuracy = %.2f%% in epoch %d;\n\
Total Time %d' % (min_loss,
                  min_loss_epoch,
                  max_accuracy,
                  max_accuracy_epoch,
                  time.time() - start), end='\n\n')
print(f'parallel_train = {parallel_train}\n\
batch_size = {batch_size}\n\
epochs = {epochs}\n\
loss_function = {criterion}\n\
optimizer = {optimizer}\n')
print(net, end='\n\n')
