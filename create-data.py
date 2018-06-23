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

2018-06-23
用K类法进行分类 ，分成5类，第二类有问题，制作1，3，4，5类

K1：
    类别         训练集数量    验证集数量       测试集数量       总数目 
    bie             1970        170             185  
    hege            2891        246             247
    huang           2196        191             209
    jixing          1865        160             175
    sgong           3217        274             302
    总数目          12139       1041            1118           14298
    
K3：
    类别         训练集数量    验证集数量       测试集数量       总数目
    bie             1983        172             172  
    hege            2889        253             249
    huang           2213        202             192
    jixing          1880        160             160
    sgong           3248        287             528
    总数目          12213       1074             1301          14588
    

K4：
    类别         训练集数量    验证集数量       测试集数量       总数目
    bie             1985        170             170  
    hege            2886        260             252
    huang           2224        192             181
    jixing          1870        158             170
    sgong           3496        301             280
    总数目          12461       1081            1053           14595

K5：
    类别         训练集数量    验证集数量       测试集数量       总数目
    bie             1984        180             171  
    hege            2891        265             247
    huang           2222        207             183
    jixing          1881        162             159
    sgong           3495        303             281
    总数目          12473       1117            1041           14631
         
'''

# im = Image.open("./test.png")
# nweimg = im.resize((227, 227), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
# nweimg.save("nweimg.png")

# cwd = "C:/Users/ww/Desktop/python tf/data-canjian-picture/"
# cwd = "E:/Python/tensorflow/data-canjian-picture/"
# cwd = "E:/Python/tensorflow/data-canjian-validation/"
cwd = "E:/Python/tensorflow/K分类数据集/dataset5/train/"
classes = ["bie","hege","huang", "jixing","sgong"]
# training = tf.python_io.TFRecordWriter("canjian-train-array.tfrecords")  # 要生成的文件
# validation = tf.python_io.TFRecordWriter("canjian-validation-array.tfrecords")
test = tf.python_io.TFRecordWriter("K5-train.tfrecords")

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
for i in range(10):
    random.shuffle(list_img)

lenth = len(list_img)

for each in range(lenth):
    # if each < 10772:
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

    # elif each >=10772 and each <12072:
    #     img = Image.open(list_img[each][0])
    #     img = img.resize((224, 224))
    #     img_raw = img.tobytes()  # 将图片转化为二进制格式
    #     label = list_img[each][1]
    #     # label = np.array(label)
    #     print(label)
    #     label = label.tobytes()
    #     example = tf.train.Example(features=tf.train.Features(feature={
    #         # 'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    #         'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
    #         'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    #     }))  # example对象对label和image数据进行封装
    #     validation.write(example.SerializeToString())  # 序列化为字符串
    #     # print(label)
    #
    # else:
    #     img = Image.open(list_img[each][0])
    #     img = img.resize((224, 224))
    #     img_raw = img.tobytes()  # 将图片转化为二进制格式
    #     label = list_img[each][1]
    #     # label = np.array(label)
    #     print(label)
    #     label = label.tobytes()
    #     example = tf.train.Example(features=tf.train.Features(feature={
    #         # 'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    #         'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
    #         'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    #     }))  # example对象对label和image数据进行封装
    #     test.write(example.SerializeToString())  # 序列化为字符串
    #     # print(label)

# training.close()
# validation.close()
test.close()
