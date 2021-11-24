import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torch
import cv2
import numpy as np


class Segmentation:
    def __init__(self):
        pass

    def __call__(self, image):
        image = np.array(image)
        return ellipse_detect(image)


def ellipse_detect(image):
    """
    椭圆肤色检测模型
    """
    skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(skinCrCbHist, (113, 155), (23, 15),
                43, 0, 360, (255, 255, 255), -1)

    YCRCB = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(YCRCB)
    skin = np.zeros(cr.shape, dtype=np.uint8)
    (x, y) = cr.shape
    for i in range(0, x):
        for j in range(0, y):
            CR = YCRCB[i, j, 1]
            CB = YCRCB[i, j, 2]
            if skinCrCbHist[CR, CB] > 0:
                skin[i, j] = 255
    dst = cv2.bitwise_and(image, image, mask=skin)
    return dst


def cr_otsu(image):
    """
    YCrCb颜色空间的Cr分量+Otsu阈值分割
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

    (y, cr, cb) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # cv2.imwrite("./imageCR.jpg", cr1)
    # cv2.imwrite("./SkinCr+OTSU.jpg", skin)

    dst = cv2.bitwise_and(image, image, mask=skin)
    # cv2.imwrite("./seperate.jpg", dst)
    return dst


def crcb_range_sceening(image):
    """
    基于YCrCb颜色空间Cr, Cb范围筛选法
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(ycrcb)

    skin = np.zeros(cr.shape, dtype=np.uint8)
    (x, y) = cr.shape
    for i in range(0, x):
        for j in range(0, y):
            if (cr[i][j] > 140) and (cr[i][j]) < 175 and (cr[i][j] > 100) and (cb[i][j]) < 120:
                skin[i][j] = 255
            else:
                skin[i][j] = 0
    # cv2.imwrite(image, image)
    # cv2.imwrite(image+"skin2 cr+cb", skin)

    dst = cv2.bitwise_and(image, image, mask=skin)
    # cv2.imwrite("cutout", dst)
    return dst


def hsv_detect(image):
    """
    HSV颜色空间H,S,V范围筛选法
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    (_h, _s, _v) = cv2.split(hsv)
    skin = np.zeros(_h.shape, dtype=np.uint8)
    (x, y) = _h.shape

    for i in range(0, x):
        for j in range(0, y):
            if(_h[i][j] > 7) and (_h[i][j] < 20) and (_s[i][j] > 28) and (_s[i][j] < 255) and (_v[i][j] > 50) and (_v[i][j] < 255):
                skin[i][j] = 255
            else:
                skin[i][j] = 0

    # cv2.imwrite(image, image)
    # cv2.imwrite(image + "hsv", skin)
    dst = cv2.bitwise_and(image, image, mask=skin)
    # cv2.imwrite("cutout", dst)
    return dst
