import cv2
import numpy as np


class Blur(object):
    """
    图像滤波

    infer: https://juejin.cn/post/6844904182143516685#heading-5
    """
    Mean = 1
    Box = 2
    Gaussian = 3
    Median = 4

    def __init__(self, kSize=(3, 3), blur=Median):
        self.kSize = kSize
        self.blur = blur

    def __call__(self, image):
        method = {
            self.Mean: self.mean,
            self.Box: self.box,
            self.Gaussian: self.gaussian,
            self.Median: self.median
        }
        image = np.array(image)
        return method[self.blur](image)

    def mean(self, image):
        """
        均值
        """
        return cv2.blur(image, self.kSize)

    def box(self, image):
        """
        方框
        """
        return cv2.boxFilter(image, -1, self.kSize, normalize=1)

    def gaussian(self, image):
        """
        高斯
        """
        return cv2.GaussianBlur(image, self.kSize, 0)

    def median(self, image):
        """
        中值
        """
        return cv2.medianBlur(image, self.kSize[0])
