import sys
import pickle
import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected

from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from skimage import feature

def convert_imagesize(images):
    tmp_images = []
    for image in images:
        tmp_image = skimage.transform.resize(image, (224, 224), mode='constant')
        tmp_images.append(tmp_image)

    return np.array(tmp_images)

def load_dataset(dataset_path):
	with open(dataset_path + '/train',mode='rb') as file:
		batch = pickle.load(file,encoding='latin1')
	
	features = batch['data'].reshape((len(batch['data']),3, 32, 32)).transpose(0, 2, 3, 1)
	labels = batch['fine_labels']

	return features, labels

def one_hot_encode(x):
    encoded = np.zeros((len(x), 100))

    for idx, val in enumerate(x):
        encoded[idx][val] = 1

    return encoded

def _preprocess_and_save(one_hot_encode, features, labels, filename):
    labels = one_hot_encode(labels)
    pickle.dump((features, labels), open(filename, 'wb'))

def preprocess(dataset_path):
	valid_features = []
	valid_labels = []
	features, labels = load_dataset(dataset_path)

	index_of_validation = int(len(features) * 0.1)

	_preprocess_and_save(one_hot_encode, features[:-index_of_validation], labels[:-index_of_validation], 'imagenet_preprocess_train.p')

    valid_features.extend(features[-index_of_validation:])
    valid_labels.extend(labels[-index_of_validation:])

    # preprocess the all stacked validation dataset
    _preprocess_and_save(one_hot_encode, np.array(valid_features), np.array(valid_labels), 'imagenet_preprocess_validation.p')

    # load the test dataset
    with open(dataset_folder_path + '/test', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # preprocess the testing data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['fine_labels']

    # Preprocess and Save all testing data
    _preprocess_and_save(one_hot_encode, np.array(test_features), np.array(test_labels), 'imagenet_preprocess_testing.p')

def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

def load_preprocess_training_batch(batch_size):
    filename = 'imagenet_preprocess_train.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    tmpFeatures = []

    for feature in features:
        tmpFeature = skimage.transform.resize(feature, (224, 224), mode='constant')
        tmpFeatures.append(tmpFeature)

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(tmpFeatures, labels, batch_size)

class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
		# return the histogram of Local Binary Patterns
		return hist

# 1st
conv1 = conv2d(input, num_outputs=96,
            kernel_size=[11,11], stride=4, padding="VALID",
            activation_fn=tf.nn.relu)
lrn1 = tf.nn.local_response_normalization(conv1, bias=2, alpha=0.0001,beta=0.75)
pool1 = max_pool2d(lrn1, kernel_size=[3,3], stride=2)

# 2nd
conv2 = conv2d(pool1, num_outputs=256,
            kernel_size=[5,5], stride=1, padding="VALID",
            biases_initializer=tf.ones_initializer(),
            activation_fn=tf.nn.relu)
lrn2 = tf.nn.local_response_normalization(conv2, bias=2, alpha=0.0001, beta=0.75)
pool2 = max_pool2d(lrn2, kernel_size=[3,3], stride=2)

#3rd
conv3 = conv2d(pool2, num_outputs=384,
            kernel_size=[3,3], stride=1, padding="VALID",
            activation_fn=tf.nn.relu)

#4th
conv4 = conv2d(conv3, num_outputs=384,
            kernel_size=[3,3], stride=1, padding="VALID",
            biases_initializer=tf.ones_initializer(),
            activation_fn=tf.nn.relu)

#5th
conv5 = conv2d(conv4, num_outputs=256,
            kernel_size=[3,3], stride=1, padding="VALID",
            biases_initializer=tf.ones_initializer(),
            activation_fn=tf.nn.relu)
pool5 = max_pool2d(conv5, kernel_size=[3,3], stride=2)

#6th
flat = flatten(pool5)
fcl1 = fully_connected(flat, num_outputs=4096,
                        biases_initializer=tf.ones_initializer(), activation_fn=tf.nn.relu)
dr1 = tf.nn.dropout(fcl1, 0.5)

#7th
fcl2 = fully_connected(dr1, num_outputs=4096,
                        biases_initializer=tf.ones_initializer(), activation_fn=tf.nn.relu)
dr2 = tf.nn.dropout(fcl2, 0.5)

#output
out = fully_connected(dr2, num_outputs=self.num_classes, activation_fn=None)