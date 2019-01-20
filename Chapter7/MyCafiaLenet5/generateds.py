import tensorflow as tf
import numpy as np
from PIL import Image
import os

image_train_path = './cifar-10/train'
label_train_path = './cifar-10/train'
tfRecord_train = './data/cifar-10_train.tfrecords'
image_test_path = './cifar-10/test'
label_test_path = './cifar-10/test'
tfRecord_test = './data/cifar-10_test.tfrecords'
data_path = './data'


def write_tfRecord(tfRecordName,image_path,label_path):
    # 从路径读入样本及标签
    # 写入对应tfRecord文件

    writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic = 0

    dirlist = os.listdir(label_path)
    for i in range(len(dirlist)):
        labels = [0] * 10
        labels[i] = 1
        Ipath = os.path.join(image_path,dirlist[i])
        for img_name in os.listdir(Ipath):
            img = Image.open(os.path.join(Ipath,img_name))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(
            feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
            }))
            writer.write(example.SerializeToString()) # SerializeToString为C++序列化API
            num_pic += 1
            print("the number of picture:", num_pic) # 显示进度

    writer.close()
    print("write tfrecord successful")


def generate_tfRecord():

    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
        print("The directory was created successfully")
    else:
        print("directory already exists")

    write_tfRecord(tfRecord_train,image_train_path,label_train_path)
    write_tfRecord(tfRecord_test,image_test_path,label_test_path)


def read_tfRecord(tfRecord_path):
    # 从文件路径读取封装好的二进制文件
    # 返回两个列表，包含全部样本和对应标签

    filename_queue = tf.train.string_input_producer([tfRecord_path], shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={'label': tf.FixedLenFeature([10], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string)})
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img.set_shape([3072])   # 图像为32*32*3=3072
    img = tf.cast(img, tf.float32) * (1. / 255) # 将RGB变为0-1之间的数
    label = tf.cast(features['label'], tf.float32)  # 将原本存为int64的标签转化

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
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=num, num_threads=2, capacity=10000,
                                                    min_after_dequeue=7000)
    return img_batch, label_batch

def main():
    generate_tfRecord()

if __name__ == '__main__':
    main()