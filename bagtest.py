import rosbag
import time
import string
import os #for file management make directory
import shutil #for file management, copy file
import cv2
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf
import numpy as np
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
            'image/timestamp': tf.FixedLenFeature([], tf.int64),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/encodedcam_right':tf.FixedLenFeature([], tf.string),
            'image/source': tf.FixedLenFeature([], tf.string),
            'image/encodedcam_center': tf.FixedLenFeature([], tf.string),
            'image/encodedcam_left': tf.FixedLenFeature([], tf.string),
            # Extract features using the keys set during creation
        }
    sample = tf.parse_single_example(data_record, features)
    return sample

listOfBagFiles = [f for f in os.listdir(".") if f[-4:] == ".bag"]
bagFile = listOfBagFiles[0]
bag = rosbag.Bag(bagFile)
bagContents = bag.read_messages()
bagName = bag.filename
listOfTopics = ['/cam_right/image_raw', '/cam_left/image_raw', '/cam_center/image_raw']

first_run = True
for topicName in listOfTopics:
    subtopic, image_msg, t = bag.read_messages(topicName).next()    # for each instant in time that has data for topicName
        #parse data from this instant, which is of the form of multiple lines of "Name: value\n"
        #    - put it in the form of a list of 2-element lists
    cv_bridge = CvBridge()
    cv_image = cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
    _, encoded = cv2.imencode('.' + 'jpg', cv_image)
    image_width = image_msg.width
    image_height = image_msg.height
    camera_name = string.split(topicName,'/')[1]
    if first_run:
        feature_dict = {
            'image/timestamp': feature_int64(image_msg.header.stamp.to_nsec()),
            'image/height': feature_int64(image_height),
            'image/width': feature_int64(image_width),
            'image/format': feature_bytes('jpg'),
            'image/encoded'+camera_name: feature_bytes(encoded.tobytes()),
            'image/source': feature_bytes(topicName),
        }
        first_run = False
    else:
        feature_dict['image/encoded'+camera_name] = feature_bytes(encoded.tobytes())

# Write TF RECORD
example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
serialized_example = example.SerializeToString()
filename = 'test.tfrecord'
# Write the `tf.Example` observations to the file.
with tf.python_io.TFRecordWriter(filename) as writer:
    writer.write(serialized_example)

# TEST TF RECORD

# Initialize all tfrecord paths
dataset = tf.data.TFRecordDataset(['test.tfrecord'])
dataset = dataset.map(extract_fn)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    try:
        while True:
            data_record = next_element
            image = sess.run(tf.image.decode_image(data_record['image/encodedcam_center']))
            print(image)

    except:
        pass
cv2.imshow('image',np.array(image))
cv2.waitKey(0)
cv2.destroyAllWindows()
