import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torch
import cv2
import numpy as np
import torchvision
from util.device import device

# model = torch.hub.load('pytorch/vision:v0.10.0',
#                        'fcn_resnet50', pretrained=True)
# or
model = torch.hub.load('pytorch/vision:v0.10.0',
                       'fcn_resnet101', pretrained=True, skip_validation=True)
model.to(device)
model.eval()
input_image = Image.open('/home/djy/dl/data/image.jpg')
# input_image = input_image.convert("RGB")
preprocess = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
# create a mini-batch as expected by the model
input_batch = input_tensor.unsqueeze(0)


with torch.no_grad():
    output = model(input_batch)['out'][0]
print(output.shape)
output_predictions = output.argmax(0)
print(output_predictions.shape)

palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output_predictions.byte().cpu().numpy()
                    ).resize(input_image.size)
r.putpalette(colors)
print(r)
plt.imshow(r)
plt.savefig('./result.jpg')
