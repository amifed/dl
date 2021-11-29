import torch
from torch._C import TracingState
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import torch
import torchvision.models as models
from models.alexnet import AlexNet
from PIL import Image

# transform = transforms.Compose(
#     [
#         # transforms.Grayscale(),
#         transforms.Resize([256, 256]),
#         # tfs.Binary(),
#         transforms.ToTensor(),
#     ])


# data_dir = '/home/djy/dataset/uni_dataset'
# dataset = torchvision.datasets.ImageFolder(
#     root=data_dir, transform=transform)
# dataloader = torch.utils.data.DataLoader(
#     dataset, batch_size=4, shuffle=True, num_workers=4)
# class_names = dataset.classes


# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     # img = inp / 2 + 0.5     # unnormalize
#     # npimg = img.numpy()
#     # plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     # plt.show()
#     inp = inp.numpy().transpose((1, 2, 0))
#     # mean = np.array([0.485, 0.456, 0.406])
#     # std = np.array([0.229, 0.224, 0.225])
#     # inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.savefig('./result.jpg')
#     # plt.pause(0.001)  # pause a bit so that plots are updated


# # Get a batch of training data
# inputs, classes = next(iter(dataloader))

# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs)

# imshow(out, title=[class_names[x] for x in classes])

# we do not specify pretrained=True, i.e. do not load default weights
classes = ('bzx', 'cwx', 'hdx', 'mtx', 'nqx', 'qtx', 'zxx')
model = AlexNet(num_classes=len(classes))
model.load_state_dict(torch.load(
    '/home/djy/dl/result/2021-11-24_21:21:44/model.pth'))

transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=(0, 180)),
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


model.eval()
image = Image.open('/home/djy/dl/data/bzx_1.jpg')
seg_image = Image.open('/home/djy/dl/data/seg_bzx_1.jpg')
image, seg_image = transform(image), transform(image)
with torch.no_grad():
    pred = model(image, image)

    _, predictions = torch.max(pred.data, 1)
    print(predictions)
    # predicted, actual = classes[pred[0].argmax(0)], classes[y]
    # print(f'Predicted: "{predicted}", Actual: "{actual}"')
