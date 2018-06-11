# -*- coding: UTF-8 -*-
import  tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def read_and_decode(filename):  # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           # 'label': tf.FixedLenFeature([], tf.int64),     #适用标签为单个数字
                                           'label': tf.FixedLenFeature([], tf.string),      #此种方法适用于标签是类似于[0,0,0,1,0,0,0]
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 将image数据和label取出来


    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])  # reshape为224*224的3通道图片
    label = tf.decode_raw(features['label'], tf.int32)  #此种方法适用于标签是类似于[0,0,0,1,0,0,0]
    label = tf.reshape(label, [5])

    '''不知道这个东西有什么作用！！！！！'''
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量
    # label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量，如果标签是单个的数字用此种方法
    return img, label

image_test, label_test = read_and_decode("canjian-test-array.tfrecords")
# image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=1, capacity=100, min_after_dequeue=20)     #乱序取出相应图片
image_test_batch, label_test_batch = tf.train.batch([image_test, label_test], batch_size=1300)

with tf.Session() as sess: #开始一个会话
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)

    for i in range(1):
        # image_batch, label_batch = sess.run([image, label])  # 在会话中取出image和label
        example, l = sess.run([image_test_batch, label_test_batch])#在会话中取出image和label

        # # 以下输出数据的类型与大小------类型：numpy.ndarray   大小：（batch_size,224,224,3）
        # print(type(example))
        # print(example.shape)

        # #以下输出标签的类型与大小------类型：numpy.ndarray   大小：（batch_size,5）
        # print(type(l))
        # print(l.shape)

        # #用于保存图片
        # img = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
        # example.save('_''Label_' + str(i) + '.jpg')  # 存下图片

        # # 用于显示图片
        # example = np.reshape(example, [224, 224, 3])
        # plt.imshow(example)
        # plt.show()
        # print(l)


    coord.request_stop()
    coord.join(threads)
