import torch
from torch.utils import data
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class experimental_dataset(Dataset):

    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data.shape[0])

    def __getitem__(self, idx):
        item = self.data[idx]
        item = self.transform(item)
        return item


transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])


x = torch.rand(8, 1, 2, 2)
print(x)

dataset = experimental_dataset(x, transform)

for item in dataset:
    print(item)
