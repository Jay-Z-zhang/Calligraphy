import tensorflow as tf
from network import Network
import matplotlib.pyplot as plt
import numpy as np
import glob
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.contrib.data import Dataset
from tensorflow.contrib.data import Iterator

VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
class_name = ['Bus', 'Microbus', 'Sedan', 'SUV', 'Truck']
validate_image_path = 'validate/'  # 指定验证集数据路径（根据实际情况指定验证数据集的路径）

x = tf.placeholder(tf.float32, [1, 224, 224, 3])
model = Network(x, 1, 5)
score = tf.nn.softmax(model.fc8)
max = tf.arg_max(score, 1)

t_num = 0
f_num = 0
image_path = np.array(glob.glob(validate_image_path + '*.jpg')).tolist()
fo = open("false.txt", "w")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "./tmp/checkpoints_t16/model_epoch7.ckpt")
    for i in range(len(image_path)):
        img_string = tf.read_file(image_path[i])
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [224, 224])
        img_resized = img_resized[:, :, ::-1]
        img_resized = np.asarray(img_resized.eval(), dtype='uint8')
        img_resized = img_resized.reshape((1, 224, 224, 3))
        prob = sess.run(max, feed_dict={x: img_resized})[0]
        t = -1
        if 'Bus' in image_path[i]:
            t = 0
        elif 'Microbus' in image_path[i]:
            t = 1
        elif 'Sedan' in image_path[i]:
            t = 2
        elif 'SUV' in image_path[i]:
            t = 3
        elif 'Truck' in image_path[i]:
            t = 4
        if t == prob:
            t_num += 1
        else:
            f_num += 1
            fo.write(image_path[i] + '_Prediction:' + str(class_name[prob]) + '\n')

print(t_num / (t_num + f_num))
