import rosbag
import time
import string
import os #for file management make directory
import shutil #for file management, copy file
import cv2, io
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# get list of only bag files in current dir.

# Parsing function
def extract_fn(data_record):
    features = {
            'timestamp': tf.FixedLenFeature([], tf.int64),
            'los': tf.FixedLenFeature([], tf.int64),
            'los_rate': tf.FixedLenFeature([], tf.int64),
            'loom': tf.FixedLenFeature([], tf.int64),
            'image_height': tf.FixedLenFeature([], tf.int64),
            'image_width': tf.FixedLenFeature([], tf.int64),
            'image/cam_right/1':tf.FixedLenFeature([], tf.string),
            'image/cam_right/2':tf.FixedLenFeature([], tf.string),
            'image/cam_right/3':tf.FixedLenFeature([], tf.string),
            'image/cam_right/4':tf.FixedLenFeature([], tf.string),
            'image/cam_right/5':tf.FixedLenFeature([], tf.string),
            'image/cam_right/6':tf.FixedLenFeature([], tf.string),
            'image/cam_center/1': tf.FixedLenFeature([], tf.string),
            'image/cam_center/2': tf.FixedLenFeature([], tf.string),
            'image/cam_center/3': tf.FixedLenFeature([], tf.string),
            'image/cam_center/4': tf.FixedLenFeature([], tf.string),
            'image/cam_center/5': tf.FixedLenFeature([], tf.string),
            'image/cam_center/6': tf.FixedLenFeature([], tf.string),
            'image/cam_left/1': tf.FixedLenFeature([], tf.string),
            'image/cam_left/2': tf.FixedLenFeature([], tf.string),
            'image/cam_left/3': tf.FixedLenFeature([], tf.string),
            'image/cam_left/4': tf.FixedLenFeature([], tf.string),
            'image/cam_left/5': tf.FixedLenFeature([], tf.string),
            'image/cam_left/6': tf.FixedLenFeature([], tf.string),
            #'label': tf.FixedLenFeature([], tf.int64),
            # Extract features using the keys set during creation
        }
    parsed_sample = tf.parse_single_example(data_record, features)

    sample = {
        'timestamp': parsed_sample['timestamp'],
        'image_height': parsed_sample['image_height'],
        'image_width': parsed_sample['image_width'],
        #'image/cam_left/6': tf.FixedLenFeature([], tf.string),
        #'label': tf.FixedLenFeature([], tf.int64),
        # Extract features using the keys set during creation
    }

    for key, value in parsed_sample.iteritems() :
        if key[0:9]=='image/cam':
            sample[key] = tf.image.decode_png(value)

    return sample




# ALEXANDRE OLHAR DAQUI PARA BAIXO APENAS

    # Initialize all tfrecord paths
dataset = tf.data.TFRecordDataset(['test5.tfrecord'])
dataset = dataset.map(extract_fn)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()


with tf.Session() as sess:
          
    data_record = next_element

    image = data_record['image/cam_left/1']
    image = sess.run(image)

    print(image)

    plt.imshow(image)
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

