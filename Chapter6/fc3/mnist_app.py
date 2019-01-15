# coding:utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward
import mnist_forward
import os

def restore_model(testPicArr):
    # 取出模型并预测
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None) # 前向传播获得预测结果
        preValue = tf.argmax(y, 1)  # 预测的结果是输出数组的最大值所对应的索引号

        # 取出模型
        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 预测结果
                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1


def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28, 28), Image.ANTIALIAS)  # 将图片转换为28*28格式，Image.ANTIALIAS表示消除锯齿
    im_arr = np.array(reIm.convert('L'))  # 将reIm转换为灰度图，生成数组
    threshold = 50  # 二色化阈值
    # 对每个像素点反色
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]  # 由黑底白字转换成白底黑字
            # 二色化处理， 滤掉噪声
            if im_arr[i][j] < threshold:
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255

    # 将图片reshape成 1行784列
    # 将像素点转换为浮点型0~1
    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0 / 255.0)
    # 返回处理过后的图片（得到了一个一维列向量）
    return img_ready


def application():
    # 原教程方法:
    # testNum = input("input the number of test pictures:")
    # for i in range(eval(testNum)):
    #     testPic = input("the path of test picture:")
    #     testPicArr = pre_pic(testPic)
    #     preValue = restore_model(testPicArr)
    #     print("The prediction number is:", preValue)

    # 提供一种识别指定文件夹下所有文件的方法： by Wh1isper

    testPicPath = input("the path of test picture:")
    dirs = os.listdir(testPicPath)
    for files in dirs:
        testPicArr = pre_pic(os.path.join(testPicPath,files))
        preValue = restore_model(testPicArr)
        print("File name:{}, The prediction number is:{}".format(repr(files.title()),preValue[0]))




def main():
    application()


if __name__ == '__main__':
    main()
