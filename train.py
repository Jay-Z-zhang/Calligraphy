import os
import numpy as np
import tensorflow as tf
from network import Network
from datagenerator import ImageDataGenerator
from datetime import datetime
import glob
from tensorflow.contrib.data import Iterator

learning_rate = 1e-4
num_epochs = 1  # 迭代次数
batch_size = 50
dropout_rate = 0.5
num_classes = 5  # 类别数量
display_step = 5

filewriter_path = "tmp/tensorboard_test"  # tensorboard文件路径
checkpoint_path = "tmp/checkpoints_test"  # 模型和参数路径

if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

train_image_path = 'train/'  # 训练集数据路径
test_image_path = 'test/'  # 测试集数据路径

label_path = []
test_label = []

# 训练集生成
image_path = np.array(glob.glob(train_image_path + '*.jpg')).tolist()
for i in range(len(image_path)):
    if 'Bus' in image_path[i]:
        label_path.append(0)
    elif 'Microbus' in image_path[i]:
        label_path.append(1)
    elif 'Sedan' in image_path[i]:
        label_path.append(2)
    elif 'SUV' in image_path[i]:
        label_path.append(3)
    elif 'Truck' in image_path[i]:
        label_path.append(4)

# 测试集生成
test_image = np.array(glob.glob(test_image_path + '*.jpg')).tolist()
for i in range(len(test_image)):
    if 'Bus' in image_path[i]:
        test_label.append(0)
    elif 'Microbus' in image_path[i]:
        test_label.append(1)
    elif 'Sedan' in image_path[i]:
        test_label.append(2)
    elif 'SUV' in image_path[i]:
        test_label.append(3)
    elif 'Truck' in image_path[i]:
        test_label.append(4)

# 调用图片生成器，把训练集图片转换成三维数组
tr_data = ImageDataGenerator(
    images=image_path,
    labels=label_path,
    batch_size=batch_size,
    num_classes=num_classes)

# 调用图片生成器，把测试集图片转换成三维数组
test_data = ImageDataGenerator(
    images=test_image,
    labels=test_label,
    batch_size=batch_size,
    num_classes=num_classes,
    shuffle=False)

with tf.name_scope('input'):
    # 定义迭代器
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)

    training_initalize = iterator.make_initializer(tr_data.data)
    testing_initalize = iterator.make_initializer(test_data.data)

    # 定义每次迭代的数据
    next_batch = iterator.get_next()

x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# 图片数据通过网络处理
model = Network(x, keep_prob, num_classes)

# 执行整个网络图
score = model.fc8

with tf.name_scope('loss'):
    # 损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))
    tf.summary.scalar('loss', loss)

with tf.name_scope('optimizer'):
    # 优化器
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 定义网络精确度
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

# 把精确度加入到Tensorboard

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(filewriter_path)
saver = tf.train.Saver()

# 定义一代的迭代次数
train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
test_batches_per_epoch = int(np.floor(test_data.data_size / batch_size))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # saver.restore(sess, "./tmp/checkpoints_t18/model_epoch10.ckpt")

    # 把模型图加入Tensorboard
    writer.add_graph(sess.graph)

    print("{} 训练开始".format(datetime.now()))
    print("{} Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))

    # 迭代所有次数
    for epoch in range(num_epochs):
        sess.run(training_initalize)
        print("{} 迭代{}次开始".format(datetime.now(), epoch + 1))

        # 开始训练每一代
        for step in range(train_batches_per_epoch):
            img_batch, label_batch = sess.run(next_batch)
            sess.run(train_op, feed_dict={x: img_batch, y: label_batch, keep_prob: dropout_rate})
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})

                writer.add_summary(s, epoch * train_batches_per_epoch + step)

        # 测试模型精确度
        print("{} 测试精度".format(datetime.now()))
        sess.run(testing_initalize)
        test_acc = 0.
        test_count = 0

        for _ in range(test_batches_per_epoch):
            img_batch, label_batch = sess.run(next_batch)
            acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.0})
            test_acc += acc
            test_count += 1

        test_acc /= test_count

        print("{} 精度 = {:.4f}".format(datetime.now(), test_acc))

        # 把训练好的模型存储起来
        print("{} 保存模型".format(datetime.now()))

        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} 迭代{}次结束".format(datetime.now(), epoch + 1), save_path)
