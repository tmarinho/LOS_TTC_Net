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

def feature_int64(value_list):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value_list, list):
        value_list = [value_list]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))


def feature_float(value_list):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value_list, list):
        value_list = [value_list]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))


def feature_bytes(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def feature_bytes_list(value_list, skip_convert=False):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value_list))


# Parsing function
def extract_fn(data_record):
    features = {
            'timestamp': tf.FixedLenFeature([], tf.int64),
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
    sample = tf.parse_single_example(data_record, features)
    return sample

def encode_fn(image_msg, sess):
    cv_image = cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
    # 
    encoded = sess.run(tf.image.encode_png(cv_image))
    #_, encoded = cv2.imencode('.' + 'png', cv_image)
    return encoded
    

listOfBagFiles = [f for f in os.listdir(".") if f[-4:] == ".bag"]
listOfTopics = ['/cam_right/image_raw', '/cam_left/image_raw', '/cam_center/image_raw']
cv_bridge = CvBridge()
filename = 'test5.tfrecord'

ready_counter = [1, 1, 1]
time_stamp_previous = 0
with tf.python_io.TFRecordWriter(filename) as writer:
    with tf.Session() as sess:
        for bagFile in listOfBagFiles:
            print(bagFile)
            bag = rosbag.Bag(bagFile)
            bagContents = bag.read_messages(topics=listOfTopics) 
            bagName = bag.filename
            feature_dict = {}
            ready = [False, False, False]
            k =0
            for topic, msg, t in bagContents:

                #subtopic, image_msg, t = bag.read_messages(topicName).next()    # for each instant in time that has data for topicName
                # if topic=='/labelesStamped' and ready[0] and ready[1] and ready[2]:
                if ready[0] and ready[1] and ready[2]:
                    feature_dict['timestamp']=feature_int64(msg.header.stamp.to_nsec())
                    feature_dict['image_width'] = feature_int64(image_width)
                    feature_dict['image_height'] = feature_int64(image_height)
                    
                    # feature_dict['label'] = [msg.x, msg.y, msg.z]
                    ready_counter = [1, 1, 1]
                    ready = [False, False, False]

                    print('Tf_record writen')
                    # Write TF RECORD
                    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                    serialized_example = example.SerializeToString()

                    # Write the `tf.Example` observations to the file.
                    writer.write(serialized_example)
                    feature_dict = { }

                
                encoded = encode_fn(msg, sess)

                if topic=='/cam_right/image_raw':
                    if ready_counter[0] <= 6:
                        feature_dict['image/cam_right/'+str(ready_counter[0])] = feature_bytes(encoded)
                        ready_counter[0] += 1
                    else:
                        ready[0]= True 

                elif topic=='/cam_left/image_raw':
                    if ready_counter[1] <=6:
                        feature_dict['image/cam_left/'+str(ready_counter[1])] = feature_bytes(encoded)
                        ready_counter[1] += 1
                    else:
                        ready[1] = True
                elif topic=='/cam_center/image_raw':
                    if ready_counter[2]  <=6:
                        feature_dict['image/cam_center/'+str(ready_counter[2])] = feature_bytes(encoded)
                        ready_counter[2] += 1
                    else:
                        ready[2] = True

                    image_width = msg.width
                    image_height = msg.height
    
    

# TEST TF RECORD

# Initialize all tfrecord paths
dataset = tf.data.TFRecordDataset(['test5.tfrecord'])
dataset = dataset.map(extract_fn)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
          
    data_record = next_element
    image = data_record['image/cam_left/1']
    image = sess.run(tf.image.decode_png(image))

    print(image)

    plt.imshow(image)
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
