from functools import total_ordering
from PIL.Image import preinit
from matplotlib import image
from matplotlib.pyplot import cla
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from models.lenet import LeNet
from models.alexnet import AlexNet
import models.vgg as vgg
import models.resnet as resnet


# use gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1. load & normalize
transform = transforms.Compose(
    [transforms.Resize((32, 32)), transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 100
trainset_path = '/home/djy/dataset/ndata100/train'
testset_path = '/home/djy/dataset/ndata100/test'

# 训练图片
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=False, transform=transform)
trainset = torchvision.datasets.ImageFolder(
    root=trainset_path, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# 测试图片
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=False, transform=transform)
testset = torchvision.datasets.ImageFolder(
    root=testset_path, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)
# test_dataiter = iter(testloader)
# test_image, test_label = test_dataiter.next()

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes = ('bzx', 'cwx', 'hdx', 'mtx', 'nqx', 'nzx', 'qtx', 'sjx', 'zxx')

# 2. define a CNN
net = resnet.resnet18(num_classes=len(classes))
net.to(device)
print(f'Train Model: LeNet, Using device {device}')


# 3. define a loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 4. train
epochs = 64
max_accuracy, max_epoch = 0, 0

for epoch in range(epochs):  # loop over the dataset multiple times

    print(f'\n========== Train Epoch {epoch + 1} ==========', end='\n')

    """train"""
    running_loss = 0.0
    for i, [inputs, labels] in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print('Loss: %.3f' %
          (running_loss / len(trainloader)), end='\t')

    """test"""
    net.eval()
    correct = 0
    total = 0
    # count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for images, labels in testloader:

            images, labels = images.to(device), labels.to(device)
            outputs = net(images)

            _, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            for label, prediction in zip(labels, predictions):
                if prediction == label:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print('Accuracy: %d %%' % (100 * correct / total))
    if ((100 * correct / total) > max_accuracy):
        max_accuracy, max_epoch = (100 * correct / total), epoch
    for classname, correct_cnt in correct_pred.items():
        accuracy = 100 * float(correct_cnt) / total_pred[classname]
        print('Accurary for class {:5s} is: {:.1f} %'.format(
            classname, accuracy), end='\n')

print('Finished Training, Max Accuracy %d %%' % (max_accuracy))
SAVE_PATH = 'LeNet.pth'
torch.save(net.state_dict(), SAVE_PATH)
