# coding:utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import os

image_train_path = './mnist_data_jpg/mnist_train_jpg_60000/'
label_train_path = './mnist_data_jpg/mnist_train_jpg_60000.txt'
tfRecord_train = './data/mnist_train.tfrecords'
image_test_path = './mnist_data_jpg/mnist_test_jpg_10000/'
label_test_path = './mnist_data_jpg/mnist_test_jpg_10000.txt'
tfRecord_test = './data/mnist_test.tfrecords'
data_path = './data'
resize_height = 28
resize_width = 28


def write_tfRecord(tfRecordName, image_path, label_path):
    # 创建二进制文件

    writer = tf.python_io.TFRecordWriter(tfRecordName)  # 实例化写入对象
    num_pic = 0 # 记录封装的图片数量

    # 读取标签文件
    f = open(label_path, 'r')
    contents = f.readlines()
    f.close()

    # 根据标签文件初始化图片路径和对应标签
    # 每次写入一个图片即对应标签
    for content in contents:
        value = content.split()
        img_path = image_path + value[0]
        img = Image.open(img_path)
        img_raw = img.tobytes()
        labels = [0] * 10
        labels[int(value[1])] = 1

        # 'img_raw'集合的特征是：比特流列表，以img_raw为输入变量写入文件
        # 'label'集合的特征是：Int64列表，以labels为输入变量写入文件
        # 上述两个集合被封装为tf.train.Features传入tf.train.Example，成为tf.train.Example
        # 再通过writer.write写入
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
            }))
        writer.write(example.SerializeToString()) # SerializeToString为C++序列化API
        num_pic += 1
        print("the number of picture:", num_pic) # 显示进度

    # 写入完毕
    writer.close()
    print("write tfrecord successful")


def generate_tfRecord():
    # 新建目录data_path，在目录下创建二进制文件

    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
        print('The directory was created successfully')
    else:
        print('directory already exists')
    write_tfRecord(tfRecord_train, image_train_path, label_train_path)
    write_tfRecord(tfRecord_test, image_test_path, label_test_path)


def read_tfRecord(tfRecord_path):
    # 从文件路径读取封装好的二进制文件
    # 返回两个列表，包含全部样本和对应标签

    filename_queue = tf.train.string_input_producer([tfRecord_path], shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={'label': tf.FixedLenFeature([10], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string)})
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img.set_shape([784])
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.float32)
    return img, label


def get_tfrecord(num, isTrain=True):
    # 顺序取出capacity个样本和对应标签
    # 其中，随机取出BATCH_SIZE大小的样本和对应标签
    # 返回两个列表，包含BATCH_SIZE大小的样本和对应标签

    if isTrain:
        tfRecord_path = tfRecord_train  # 选择训练集
    else:
        tfRecord_path = tfRecord_test   # 选择测试集
    img, label = read_tfRecord(tfRecord_path)
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=num, num_threads=2, capacity=1000,
                                                    min_after_dequeue=700)
    return img_batch, label_batch


def main():
    generate_tfRecord()


if __name__ == '__main__':
    main()
