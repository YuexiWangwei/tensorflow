# -*- coding: UTF-8 -*-
import tensorflow as tf
import datetime

test = "canjian-test-array.tfrecords"
bie  = "canjian-test-bie-array.tfrecords"
hege = "canjian-test-hege-array.tfrecords"
huang = "canjian-test-huang-array.tfrecords"
jixing = "canjian-test-jixing-array.tfrecords"
sgong = "canjian-test-sgong-array.tfrecords"

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename],shuffle=True)  # 生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 将image数据和label取出来

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])  # reshape为224*224的3通道图片
    img = tf.cast(img, tf.float32) * (1. / 255)  # 在流中抛出img张量
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量
    label = tf.decode_raw(features['label'], tf.int32)  # 此种方法适用于标签是类似于[0,0,0,1,0,0,0]
    label = tf.reshape(label, [5])

    return img, label

def weight_varialves(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_vairables(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

myGraph = tf.Graph()
with myGraph.as_default():
    with myGraph.name_scope("Inputs"):
        x_raw = tf.placeholder(tf.float32, shape=[None, 224,224,3])     #50176=224*224
        y = tf.placeholder(tf.float32, shape=[None, 5])

    #每一个卷积层包含卷积、激活、池化   第一层卷积完成以后得到112*112*64大小的数据
    with myGraph.name_scope("conv1"):
        # x = tf.reshape(x_raw, shape=[-1, 224, 224, 3])
        w_conv1 = weight_varialves([3,3,3,64])
        b_conv1 = bias_vairables([64])
        out_conv1 = tf.nn.relu(tf.nn.conv2d(x_raw,w_conv1,strides=[1,1,1,1],padding="SAME") + b_conv1)
        out_pool1 = tf.nn.max_pool(out_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")


    #第二层卷积完成以后得到56*56*128大小的数据
    with myGraph.name_scope("conv2"):
        w_conv2 = weight_varialves([3,3,64,128])
        b_conv2 = bias_vairables([128])
        out_conv2 = tf.nn.relu(tf.nn.conv2d(out_pool1,w_conv2,strides=[1,1,1,1],padding="SAME") + b_conv2)
        out_pool2 = tf.nn.max_pool(out_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    # 第三层卷积完成以后得到28*28*192大小的数据
    with myGraph.name_scope("conv3"):
        w_conv3 = weight_varialves([3,3,128,192])
        b_conv3 = bias_vairables([192])
        out_conv3 = tf.nn.relu(tf.nn.conv2d(out_pool2,w_conv3,strides=[1,1,1,1],padding="SAME") + b_conv3)
        out_pool3 = tf.nn.max_pool(out_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    # 第四层卷积完成以后得到14*14*192大小的数据
    with myGraph.name_scope("conv4"):
        w_conv4 = weight_varialves([3,3,192,192])
        b_conv4 = bias_vairables([192])
        out_conv4 = tf.nn.relu(tf.nn.conv2d(out_pool3,w_conv4,strides=[1,1,1,1],padding="SAME") + b_conv4)
        out_pool4 = tf.nn.max_pool(out_conv4,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    # 第一层全连接完成以后得到[1,4096]大小的数据
    with myGraph.name_scope("fc1"):
        w_fc1 = weight_varialves([14*14*192,4096])
        b_fc1 = bias_vairables([4096])
        out_pool4_flat = tf.reshape(out_pool4,[-1,14*14*192])
        out_fc1 = tf.nn.relu(tf.matmul(out_pool4_flat,w_fc1) + b_fc1)
        keep_prob = tf.placeholder(tf.float32)
        out_fc1_drop = tf.nn.dropout(out_fc1, keep_prob)

    # 第二层全连接完成以后得到[1,5]大小的数据
    with myGraph.name_scope("fc2"):
        w_fc2 = weight_varialves([4096,5])
        b_fc2 = bias_vairables([5])
        y_conv = tf.matmul(out_fc1_drop,w_fc2) + b_fc2

    with myGraph.name_scope("train"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y))
        result = tf.argmax(y_conv,1)
        prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32))


    ''''''"bie", "hege", "huang", "jixing", "sgong"''''''

    image_test, label_test = read_and_decode(sgong)
    image_test_batch, label_test_batch = tf.train.batch([image_test, label_test], batch_size=25)

with tf.Session(graph=myGraph) as sess:
    saver = tf.train.Saver()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # saver.restore(sess, "./cjmodel/-299.data-00000-of-00001")  # restore参数
    saver.restore(sess, "./cjmodel/second-600")
    print("restore the data!")


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print("start!")

    #用验证数据来验证正确率
    test_accuracy = 0
    result_label = []
    for i in range(8):
        test_batch = sess.run([image_test_batch, label_test_batch])
        temp_accuracy = accuracy.eval(feed_dict={x_raw: test_batch[0], y: test_batch[1], keep_prob: 1.0})
        test_accuracy = test_accuracy + temp_accuracy
        label = result.eval(feed_dict={x_raw: test_batch[0], y: test_batch[1], keep_prob: 1.0})

        #以下程序用来输出每类识别的具体情况
        for each in label:
            result_label.append(each)
    print("合格茧:%s"%result_label.count(1))
    print("畸形茧:%s"%result_label.count(3))
    print("瘪  茧:%s"%result_label.count(0))
    print("黄斑茧:%s"%result_label.count(2))
    print("双宫茧:%s"%result_label.count(4))

    test_accuracy = test_accuracy/8
    print('the accuracy is: %g' % test_accuracy)

    coord.request_stop()
    coord.join(threads)