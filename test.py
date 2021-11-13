import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import util.transforms as tfs


transform = transforms.Compose(
    [
        # transforms.Grayscale(),
        transforms.Resize([256, 256]),
        tfs.Blur(),
        # tfs.Binary(),
        tfs.Segmentation(),
        transforms.ToTensor(),
    ])


data_dir = '/home/djy/dataset/uni_dataset'
dataset = torchvision.datasets.ImageFolder(
    root=data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=4)
class_names = dataset.classes


def imshow(inp, title=None):
    """Imshow for Tensor."""
    # img = inp / 2 + 0.5     # unnormalize
    # npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.savefig('./result.jpg')
    # plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
