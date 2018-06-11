# -*- coding: UTF-8 -*-
import tensorflow as tf
import os
from PIL import Image
import random
import numpy as np

'''
注意下面的
label_raw = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
这是将标签做成类似于mnist标签的做法，这样可以仿照mnist的方式进行训练
将数组作为标签写入label时，需要将数组转化为btyes，在读数据的时候需要进行解码，即decode_raw()方法
'''

'''
图片信息：
    类别         数量      标签
    bie         2179        [1,0,0,0,0]
    hege        3148        [0,1,0,0,0]      
    huang       2433        [0,0,1,0,0]
    jixing      2057        [0,0,0,1,0]       
    sgong       3555        [0,0,0,0,1]   


    
总共：13372 - 15 -2 - 55 = 13300(还没有做！)
'''

# im = Image.open("./test.png")
# nweimg = im.resize((227, 227), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
# nweimg.save("nweimg.png")

# cwd = "C:/Users/ww/Desktop/python tf/data-canjian-picture/"
cwd = "E:/Python/tensorflow/data-canjian-picture/"
classes = ["bie","hege","huang", "jixing","sgong"]
training = tf.python_io.TFRecordWriter("canjian-train-array.tfrecords")  # 要生成的文件
validation = tf.python_io.TFRecordWriter("canjian-validation-array.tfrecords")
test = tf.python_io.TFRecordWriter("canjian-test-array.tfrecords")

# classes = ["bie","hege","huang", "jixing","sgong"]
# cwd = "C:/Users/ww/Desktop/python tf/data-canjian-test/"
# writer = tf.python_io.TFRecordWriter("canjian-test-array.tfrecords")

label_raw = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
list_img = []

for index, name in enumerate(classes):
    class_path = cwd + name + "/"
    for img_name in os.listdir(class_path):
        tuple_temp = []
        img_path = class_path + img_name  # 每一个图片的地址
        tuple_temp.append(img_path)
        # tuple_temp.append(index)
        tuple_temp.append(label_raw[index])     #将数组标签写入临时列表
        list_img.append(tuple_temp)


#随机打乱5次，为是进行充分的shuffle
for i in range(5):
    random.shuffle(list_img)

lenth = len(list_img)

for each in range(lenth):
    if each < 10772:
        img = Image.open(list_img[each][0])
        img = img.resize((224, 224))
        img_raw = img.tobytes()  # 将图片转化为二进制格式
        label = list_img[each][1]
        # label = np.array(label)
        print(label)
        label = label.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            # 'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))  # example对象对label和image数据进行封装
        training.write(example.SerializeToString())  # 序列化为字符串
        # print(label)

    elif each >=10772 and each <12072:
        img = Image.open(list_img[each][0])
        img = img.resize((224, 224))
        img_raw = img.tobytes()  # 将图片转化为二进制格式
        label = list_img[each][1]
        # label = np.array(label)
        print(label)
        label = label.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            # 'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))  # example对象对label和image数据进行封装
        validation.write(example.SerializeToString())  # 序列化为字符串
        # print(label)

    else:
        img = Image.open(list_img[each][0])
        img = img.resize((224, 224))
        img_raw = img.tobytes()  # 将图片转化为二进制格式
        label = list_img[each][1]
        # label = np.array(label)
        print(label)
        label = label.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            # 'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))  # example对象对label和image数据进行封装
        test.write(example.SerializeToString())  # 序列化为字符串
        # print(label)

training.close()
validation.close()
test.close()
