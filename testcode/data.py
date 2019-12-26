import os
import os.path
import sys
import numpy as np
import random
from collections import OrderedDict
from skimage import io, transform, color
from skimage.feature import greycomatrix, greycoprops, daisy
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from matplotlib.gridspec import GridSpec
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import RidgeClassifier
from keras import models, layers
from keras.utils.np_utils import to_categorical

# two lists that are used to store images and their labels respectively
resized_imgs = list()

# 0:left 1:left_str 2:dc 3:tj
label_of_imgs = list()

left_dic, left_str_dic, dc_dic, tj_dic = {}, {}, {}, {}


def load_img(local_file, img_dic_file):
    for img_file in local_file:
        try:
            img = io.imread(img_file)
            if True:
                img_dic_file[os.path.basename(img)] = img
        except Exception:
            print('Error')


load_img('C://Users/Administrator/Desktop/Workspace/python/calligraphy/data/test.jpg', left_dic )


def adjust_img_size_label(img_dic_name, label, classname):
    for value in img_dic_name.values():
        resized_img = transform.resize(value, (300, 400, value.shape[-1]), preserve_range=True)
        resized_imgs.append(resized_img)
        label_of_imgs.append(label)
    print("{0} resize complete".format(classname))


adjust_img_size_label(left_dic, 0, "left")
