"""
对数据集中的图像进行分割
"""
from PIL import Image
import os
import cv2
import numpy as np
from numpy.core.numeric import outer
from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt
import gc
gc.collect()
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def YCrCb_HSV(source, target):
    img = cv2.imread(source)

    # converting from gbr to hsv color space
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # skin color range for hsv color space
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))
    HSV_mask = cv2.morphologyEx(
        HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # skin color range for hsv color space
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
    YCrCb_mask = cv2.morphologyEx(
        YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # merge skin detection (YCbCr and hsv)
    global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
    global_mask = cv2.medianBlur(global_mask, 3)
    global_mask = cv2.morphologyEx(
        global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

    HSV_result = cv2.bitwise_not(HSV_mask)
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)
    global_result = cv2.bitwise_not(global_mask)

    cv2.imwrite(target, global_result)


def deeplabv3(source, target):
    # def wrapper(source, target):
    model.eval()
    input_image = Image.open(source)
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize([640, 640]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)

    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)

    # move the input and model to GPU for speed if available
    input_batch = input_batch.to(device)

    # with torch.no_grad():
    output = model(input_batch)['out']
    output = output.squeeze()
    output_predictions = output.argmax(0)
    print(np.unique(output_predictions.detach().cpu().numpy()))
    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()
                        ).resize(input_image.size)
    r.putpalette(colors)
    r = r.convert('RGB')
    cv2.imwrite(target, np.array(r))
    # plt.imshow(r)
    # plt.savefig(target)
    # return wrapper


def segment(source, target, method):
    if not os.path.exists(source):
        raise ValueError('source not exist')
    if not os.path.exists(target):
        os.mkdir(target)
    for classname in os.listdir(source):
        origin, dist = os.path.join(
            source, classname), os.path.join(target, classname)
        if not os.path.exists(dist):
            os.mkdir(dist)
        for filename in os.listdir(origin):
            if not filename.endswith('.jpg') and not filename.endswith('.jpeg'):
                continue

            method(os.path.join(origin, filename),
                   os.path.join(dist, filename))
            torch.cuda.empty_cache()
            print(f'processing {filename}')


if __name__ == '__main__':
    model = torch.hub.load('pytorch/vision:v0.10.0',
                           'deeplabv3_resnet50', pretrained=True)
    model.to(device)
    torch.cuda.empty_cache()
    source = '/home/djy/dataset/dataset1'
    target = '/home/djy/dataset/ycrcb_hsv_dataset1'
    segment(source, target, YCrCb_HSV)
