from posixpath import join
import cv2
from os import path, listdir
import numpy as np


def crop_image(img, x, y, w, h):
    return img[y:y+h, x:x+w]


def rotate_image(img, angle, crop=False):
    """
    angle: 旋转的角度
    crop: 是否需要进行裁剪，布尔向量
    """
    w, h = img.shape[:2]
    # 旋转角度的周期是360°
    angle %= 360
    # 计算仿射变换矩阵
    M_rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    # 得到旋转后的图像
    img_rotated = cv2.warpAffine(img, M_rotation, (w, h))
    # 去除黑边的操作，定义裁切函数，后续裁切黑边使用
    # 如果需要去除黑边
    if crop:
        # 裁剪角度的等效周期是180°
        angle_crop = angle % 180
        if angle > 90:
            angle_crop = 180 - angle_crop
        # 转化角度为弧度
        theta = angle_crop * np.pi / 180
        # 计算高宽比
        hw_ratio = float(h) / float(w)
        # 计算裁剪边长系数的分子项
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * np.tan(theta)

        # 计算分母中和高宽比相关的项
        r = hw_ratio if h > w else 1 / hw_ratio
        # 计算分母项
        denominator = r * tan_theta + 1
        # 最终的边长系数
        crop_mult = numerator / denominator

        # 得到裁剪区域
        w_crop = int(crop_mult * w)
        h_crop = int(crop_mult * h)
        x0 = int((w - w_crop) / 2)
        y0 = int((h - h_crop) / 2)

        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)
    return img_rotated


def rotate90(img):
    """
    90度旋转
    """
    return rotate_image(img, 90, False)


def rotate180(img):
    """
    180度旋转
    """
    return rotate_image(img, 180, False)


def flip_vertical(img):
    """
    垂直翻转
    """
    return cv2.flip(img, 0)


def flip_horizontal(img):
    """
    水平翻转
    """
    return cv2.flip(img, 1)


def batch_augment(source, aug_dict):
    folder, filename = path.split(source)
    name, suffix = filename.split('.')
    img = cv2.imread(source)
    for k in aug_dict:
        cv2.imwrite(
            path.join(folder, f'{name}_{k}.{suffix}'), aug_dict[k](img))


def augment(source, aug_dict):
    for classname in listdir(source):
        origin = path.join(source, classname)
        files = listdir(origin)
        for filename in files:
            if not filename.endswith('.jpg') and not filename.endswith('.jpeg') and not filename.endswith('.png'):
                continue
            batch_augment(path.join(origin, filename), aug_dict)
            print(f'processing {filename}')


if __name__ == '__main__':
    aug_dict = {
        'r20': lambda img: rotate_image(img, 20, True),
        'r90': rotate90,
        'fv': flip_vertical,
        'fh': flip_horizontal
    }
    augment('/home/djy/dataset/dataset2_aug', aug_dict=aug_dict)
