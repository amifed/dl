import torch
import torchvision
from torchvision.transforms import transforms

transform = transforms.Compose(
    [transforms.Resize((32, 32)), transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# 训练图片
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=False, transform=transform)
dataset = torchvision.datasets.ImageFolder(
    root='/home/djy/dataset/dataset', transform=transform)

print(dataset)
full_size = len(dataset)
train_size = int(0.8 * full_size)
test_size = full_size - train_size
trainset, testset = torch.utils.data.random_split(
    dataset, [train_size, test_size])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)
print(trainloader, testloader)
