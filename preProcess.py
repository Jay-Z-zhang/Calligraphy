from PIL import Image, ImageOps
import cv2
import numpy as np
import matplotlib as plt


def resize_filter_gray_img(img: 'np.ndarray'):
    # resize
    res = cv2.resize(img, (400, 400), interpolation=cv2.INTER_CUBIC)
    '''
    中值滤波blur
    cv2.blur(img, (3, 3))  进行均值滤波
    参数说明：img表示输入的图片， (3, 3) 表示进行均值滤波的方框大小
    '''
    blur = cv2.blur(res, (2, 1))
    # guass = cv2.GaussianBlur(res,(1, 1),1)
    # 灰度图像
    gray_img = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    return gray_img


def binary_inverse_img(grayImg):
    # 二值化， ret 不可以省略
    ret, binImage = cv2.threshold(grayImg, 80, 255, cv2.THRESH_BINARY_INV)
    # get the height, width of img , img is (400,400)
    height, width = binImage.shape

    dst = np.zeros((height, width), np.uint8)
    # 反色计算
    for i in range(0, height):
        for j in range(0, width):
            grayPixel = binImage[i, j]
            dst[i, j] = 255 - grayPixel
    return dst


img = cv2.imread('5.jpg')
gray_img = resize_filter_gray_img(img)
processed_img = binary_inverse_img(gray_img)


cv2.imshow('binary_img1', processed_img)

cv2.waitKey(0)
