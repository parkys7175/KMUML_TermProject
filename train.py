from CNNet import CNNet
import tensorflow as tf
import sys, os, glob
import cv2
import numpy as np
import random


max_epochs = 25
base_image_path = "./DataSet/training"
base_test_path = "./DataSet/test"
image_types = ["red", "green", "yellow"]
input_img_x = 64
input_img_y = 64
l_rate = 0.01

batch_size = 32


train_full_set = []

for im_type in image_types :
    for ex in glob.glob(os.path.join(base_image_path, im_type, "*")) :
        img = cv2.imread(ex)
        if not img is None :
            im = cv2.resize(img, (input_img_x, input_img_y))
            one_hot_array = [0] * len(image_types)
            one_hot_array[image_types.index(im_type)] = 1

            train_full_set.append((im, one_hot_array, ex))


random.shuffle(train_full_set)


train_set_offset = len(train_full_set) % batch_size

train_full_set = train_full_set[ : len(train_full_set) - train_set_offset]

train_img, train_label, _ = zip(*train_full_set)



sess = tf.Session()

cnn = CNNet(sess, [input_img_x, input_img_y, 3], 3)

cnt = 0
for i in range(0, max_epochs) :
    for t in range(0, (len(train_img) // batch_size)) :
        start_batch = t * batch_size
        end_batch = (t + 1) * batch_size

        c, a, _ = cnn.train(train_img[start_batch:end_batch], train_label[start_batch:end_batch]) 

        cnn.write_summary(train_img[start_batch:end_batch], train_label[start_batch:end_batch])

        if cnt % 100 == 0 :
            print("cost : ", c, "   accu : ", a)
            cnt = 0

    # print("write summary done")

