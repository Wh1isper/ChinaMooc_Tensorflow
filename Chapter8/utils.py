#!/usr/bin/python
# coding:utf-8
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 正常显示正负号


def load_image(path):
    fig = plt.figure("Centre and Resize")
    img = io.imread(path)
    img = img / 255.0  # 归一化

    ax0 = fig.add_subplot(131)  # 131代表一行三列第一个
    ax0.set_xlabel(u'Original Picture')
    ax0.imshow(img)  # 原图

    # 取最短边，以中心为准裁剪成正方形
    short_edge = min(img.shape[:2])
    y = (img.shape[0] - short_edge) // 2
    x = (img.shape[1] - short_edge) // 2
    crop_img = img[y:y + short_edge, x:x + short_edge]

    ax1 = fig.add_subplot(132)
    ax1.set_xlabel(u"Centre Picture")
    ax1.imshow(crop_img)  # 裁剪后的图

    # 将裁剪后的图变为224*224
    re_img = transform.resize(crop_img, (224, 224))

    ax2 = fig.add_subplot(133)
    ax2.set_xlabel(u"Resize Picture")
    ax2.imshow(re_img)  # 输入神经网络的图

    img_ready = re_img.reshape((1, 224, 224, 3))

    return img_ready


def percent(value):
    return '%.2f%%' % (value * 100)
