"""
对数据集中的图像进行分割
"""
from sys import flags
from PIL import Image
import os
import cv2
import numpy as np
from numpy.lib.function_base import sort_complex

source = '/home/djy/dataset/uni_dataset'
target = '/home/djy/dataset/seg_dataset'


def segment(img, pth, num):
    cv2.imwrite(os.path.join(pth, f'{num}.jpg'), img)
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

    # show results
    # cv2.imshow("1_HSV.jpg",HSV_result)
    # cv2.imshow("2_YCbCr.jpg",YCrCb_result)
    # cv2.imshow("3_global_result.jpg",global_result)
    # cv2.imshow("Image.jpg",img)
    # cv2.imwrite(os.path.join(pth, f'{num}_HSV.jpg'), HSV_result)
    # cv2.imwrite(os.path.join(pth, f'{num}_YCbCr.jpg'), YCrCb_result)
    # cv2.imwrite(os.path.join(pth, f'{num}_global_result.jpg'), global_result)
    cv2.imwrite(os.path.join(pth, f'{num}.jpg'), global_result)


classes = ('bzx', 'cwx', 'hdx', 'mtx', 'nqx', 'qtx', 'zxx')
mp = {cls: 0 for cls in classes}

for cls in classes:
    fr, to = os.path.join(source, cls), os.path.join(target, cls)
    if not os.path.exists(to):
        os.mkdir(to)
    for file in os.listdir(fr):
        if not file.endswith('.jpg'):
            continue
        image = cv2.imread(os.path.join(fr, file))
        segment(image, to, mp[cls])
        mp[cls] += 1
