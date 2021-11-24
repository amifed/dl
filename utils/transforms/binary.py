import cv2
import numpy as np


class Binary(object):
    def __init__(self):  # ...是要传入的多个参数
        # 对多参数进行传入
        # 如 self.p = p 传入概率
        # ...
        pass

    def __call__(self, image):  # __call__函数还是只有一个参数传入
        image = np.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary


def binary_1(image):
    """
    灰度 -> 二值化
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def binary_2(image):
    """
    高斯模糊降噪 -> 二值化
    """
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)  # 先高斯模糊再二值化
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def binary_3(image):
    """
    均值迁移 -> 二值化
    """
    blurred = cv2.pyrMeanShiftFiltering(image, 10, 100)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary
