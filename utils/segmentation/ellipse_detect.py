import cv2
import numpy as np


def ellipse_detect(image):
    """
    :param image: 图片路径
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


if __name__ == '__main__':
    ellipse_detect('./data/image.jpg')
