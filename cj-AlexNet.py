# -*- coding: UTF-8 -*-
import tensorflow as tf
import datetime

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

        tf.summary.histogram("w_conv1",w_conv1)
        tf.summary.histogram("b_conv1",b_conv1)

    #第二层卷积完成以后得到56*56*128大小的数据
    with myGraph.name_scope("conv2"):
        w_conv2 = weight_varialves([3,3,64,128])
        b_conv2 = bias_vairables([128])
        out_conv2 = tf.nn.relu(tf.nn.conv2d(out_pool1,w_conv2,strides=[1,1,1,1],padding="SAME") + b_conv2)
        out_pool2 = tf.nn.max_pool(out_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

        tf.summary.histogram("w_conv2", w_conv2)
        tf.summary.histogram("b_conv2", b_conv2)

    # 第三层卷积完成以后得到28*28*192大小的数据
    with myGraph.name_scope("conv3"):
        w_conv3 = weight_varialves([3,3,128,192])
        b_conv3 = bias_vairables([192])
        out_conv3 = tf.nn.relu(tf.nn.conv2d(out_pool2,w_conv3,strides=[1,1,1,1],padding="SAME") + b_conv3)
        out_pool3 = tf.nn.max_pool(out_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

        tf.summary.histogram("w_conv3", w_conv3)
        tf.summary.histogram("b_conv3", b_conv3)

    # 第四层卷积完成以后得到14*14*192大小的数据
    with myGraph.name_scope("conv4"):
        w_conv4 = weight_varialves([3,3,192,192])
        b_conv4 = bias_vairables([192])
        out_conv4 = tf.nn.relu(tf.nn.conv2d(out_pool3,w_conv4,strides=[1,1,1,1],padding="SAME") + b_conv4)
        out_pool4 = tf.nn.max_pool(out_conv4,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

        tf.summary.histogram("w_conv4", w_conv4)
        tf.summary.histogram("b_conv4", b_conv4)

    # 第一层全连接完成以后得到[1,4096]大小的数据
    with myGraph.name_scope("fc1"):
        w_fc1 = weight_varialves([14*14*192,4096])
        b_fc1 = bias_vairables([4096])
        out_pool4_flat = tf.reshape(out_pool4,[-1,14*14*192])
        out_fc1 = tf.nn.relu(tf.matmul(out_pool4_flat,w_fc1) + b_fc1)
        keep_prob = tf.placeholder(tf.float32)
        out_fc1_drop = tf.nn.dropout(out_fc1, keep_prob)

        tf.summary.histogram("w_fc1",w_fc1)
        tf.summary.histogram("b_fc1",b_fc1)

    # 第二层全连接完成以后得到[1,5]大小的数据
    with myGraph.name_scope("fc2"):
        w_fc2 = weight_varialves([4096,5])
        b_fc2 = bias_vairables([5])
        y_conv = tf.matmul(out_fc1_drop,w_fc2) + b_fc2

        tf.summary.histogram("w_fc2",w_fc2)
        tf.summary.histogram("b_fc2",b_fc2)

    with myGraph.name_scope("train"):
        # loss = -tf.reduce_sum(y*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
        # loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y))
        # loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y)
        # train_step = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.9).minimize(loss)
        train_step = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
        prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1))
        # prediction = tf.equal(y_conv,y)
        accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32))

        tf.summary.scalar("loss",loss)
        tf.summary.scalar("accuracy",accuracy)

    # def getbatch(batchsize):
    #     print("I am here")
    #     image_batch, label_batch = tf.train.shuffle_batch([image_train, label_train], batch_size=batchsize, capacity=500,
    #                                                       min_after_dequeue=400)
    #     return image_batch, label_batch

    image_train, label_train = read_and_decode("canjian-train-array.tfrecords")
    image_validation, label_validation = read_and_decode("canjian-validation-array.tfrecords")
    # image_test, label_test = read_and_decode("canjian-test-array.tfrecords")
    # image_batch, label_batch = tf.train.shuffle_batch([image_train, label_train], batch_size=16, capacity=100, min_after_dequeue=50)
    image_train_batch, label_train_batch = tf.train.batch([image_train, label_train], batch_size=32)
    image_validation_batch, label_validation_batch = tf.train.batch([image_validation, label_validation], batch_size=32)
    # image_test_batch, label_test_batch = tf.train.batch([image_test, label_test], batch_size=1)


with tf.Session(graph=myGraph) as sess:
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()

    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./cjEvent', graph=sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print("start!")
    for epochs in range(10):

        print("%s   epoch"%(epochs+1))
        start_time = datetime.datetime.now()

        #一个周期的迭代总数
        for iteration in range(20):#13372/16==835  13372/256 == 52
            train_batch = sess.run([image_train_batch, label_train_batch])
            sess.run(train_step, feed_dict={x_raw: train_batch[0], y: train_batch[1], keep_prob: 0.5})

            # print(batch[0].shape,batch[1].shape)
            # print(x_raw.shape)
            # print(out_pool1.shape)
            # print(out_pool2.shape)
            # print(out_pool3.shape)
            # print(out_pool4_flat.shape)
            # print(out_fc1_drop.shape)
            # print(y_conv.eval(feed_dict={x_raw: train_batch[0], y: train_batch[1], keep_prob: 1.0}))
            # print(y_conv.shape)
            # print(loss1.eval(feed_dict={x_raw: batch[0], y: batch[1], keep_prob: 1.0}))
            # print(loss.eval(feed_dict={x_raw: batch[0], y: batch[1], keep_prob: 1.0}))

        iteration_time = datetime.datetime.now()

        # # 用训练数据来测试正确率
        # test_accuracy = accuracy.eval(feed_dict={x_raw: train_batch[0], y: train_batch[1], keep_prob: 1.0})
        # test_loss = loss.eval(feed_dict={x_raw: train_batch[0], y: train_batch[1], keep_prob: 1.0})
        # print("ecah epoch time: %s" % (end_time - each_epoch_time))
        # print('the accuracy is: %g' % test_accuracy)
        # print('the    loss  is: %g' % test_loss)
        #
        # print("")

        #用验证数据来验证正确率
        temp_accuracy = 0
        for i in range(40):
            validation_batch = sess.run([image_validation_batch, label_validation_batch])
            test_accuracy = accuracy.eval(feed_dict={x_raw: validation_batch[0], y: validation_batch[1], keep_prob: 1.0})
            print(test_accuracy)
            temp_accuracy = test_accuracy + temp_accuracy

        temp_accuracy = temp_accuracy/40
        test_loss = loss.eval(feed_dict={x_raw: train_batch[0], y: train_batch[1], keep_prob: 1.0})

        test_time = datetime.datetime.now()

        #跟踪参数变化
        summary = sess.run(merged, feed_dict={x_raw: train_batch[0], y: train_batch[1], keep_prob: 1.0})
        summary_writer.add_summary(summary,(epochs+1))

        summary_time = datetime.datetime.now()

        print('the accuracy is: %g' % temp_accuracy)
        print('the    loss  is: %g' % test_loss)
        print("")
        print("iteration time: %s"%(iteration_time-start_time))
        print("test      time: %s"%(test_time - iteration_time))
        print("summary   time: %s"%(summary_time - test_time))
        print("one epoch time: %s"%(summary_time - start_time))
        print("-----------------------------------")


    #保存模型，关闭线程
    # saver.save(sess,save_path='./cjmodel/',global_step=1)
    #
    # end_time = datetime.datetime.now()
    # print("total time:%s"%(end_time-each_epoch_time))

    coord.request_stop()
    coord.join(threads)


