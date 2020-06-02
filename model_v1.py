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
	with open(dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

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
	 n_batches = 5
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cifar10_batch(dataset_folder_path, batch_i)

        index_of_validation = int(len(features) * 0.1)
        _preprocess_and_save(one_hot_encode, 
                            features[:-index_of_validation], labels[:-index_of_validation], 'preprocess_batch_' + str(batch_i) + '.p')

        valid_features.extend(features[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])

    # preprocess the all stacked validation dataset
    _preprocess_and_save(one_hot_encode,
                         np.array(valid_features), np.array(valid_labels),
                         'preprocess_validation.p')

    # load the test dataset
    with open(dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # preprocess the testing data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    # Preprocess and Save all testing data
    _preprocess_and_save(one_hot_encode,
                         np.array(test_features), np.array(test_labels),
                         'preprocess_testing.p')

def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

def load_preprocess_training_batch(batch_size):
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    tmpFeatures = []

    for feature in features:
        tmpFeature = skimage.transform.resize(feature, (224, 224), mode='constant')
        tmpFeatures.append(tmpFeature)

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(tmpFeatures, labels, batch_size)

#class LocalBinaryPatterns:
#	def __init__(self, numPoints, radius):
		# store the number of points and radius
#		self.numPoints = numPoints
#		self.radius = radius
#	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
#		lbp = feature.local_binary_pattern(image, self.numPoints,
#			self.radius, method="uniform")
#		(hist, _) = np.histogram(lbp.ravel(),
#			bins=np.arange(0, self.numPoints + 3),
#			range=(0, self.numPoints + 2))
		# normalize the histogram
#		hist = hist.astype("float")
#		hist /= (hist.sum() + eps)
		# return the histogram of Local Binary Patterns
#		return hist

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

class objectLocalization:
    def __init__(self, dataset, learning_rate):
        self.dataset = dataset

        self.num_classes = 10

        self.learning_rate = learning_rate
        self.input = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input')
        self.label = tf.placeholder(tf.int32, [None, self.num_classes], name='label')

        self.logits = self.load_model()
        self.model = tf.identity(self.logits, name='logits')

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.label), name='cost')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='adam').minimize(self.cost)

        self.correct_pred = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name='accuracy')

    def load_model(self):
        # 1st
        conv1 = conv2d(self.input, num_outputs=96,
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
        return out

    def label_to_name():
        return ['fish', 'goldFish', 'shark', 'blueShark', 'hamerhead', 'stingray', 'bird', 'blueBird', 'orangeBird', 'lizard']

    def test(self, image, save_model_path):
        resize_images = []
        loaded_graph = tf.Graph()

        with tf.Session(graph=loaded_graph) as sess:
            loader = tf.train.import_meta_graph(save_model_path + '.meta')
            loader.restore(sess, save_model_path)

            loaded_x = loaded_graph.get_tensor_by_name('input:0')
            loaded_y = loaded_graph.get_tensor_by_name('label:0')
            loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
            loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

            resize_image = skimage.transform.resize(image, (224, 224), mode='constant')
            resize_images.append(resize_image)

            predictions = sess.run(
                tf.nn.softmax(loaded_logits),
                feed_dict={loaded_x: tmpTestFeatures, loaded_y: random_test_labels})

            label_names = load_label_names()

            predictions_array = []
            pred_names = []

            for index, pred_value in enumerate(predictions[0]):
                tmp_pred_name = label_names[index]
                predictions_array.append({tmp_pred_name : pred_value})

            return predictions_array

    def train_from_ckpt(self, epochs, batch_size, valid_set, save_model_path):
        tmpValidFeatures, valid_labels = valid_set

        loaded_graph = tf.Graph()

        with tf.Session(graph=loaded_graph) as sess:
            loader = tf.train.import_meta_graph(save_model_path + '.meta')
            loader.restore(sess, save_model_path)

            loaded_x = loaded_graph.get_tensor_by_name('input:0')
            loaded_y = loaded_graph.get_tensor_by_name('label:0')
            loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
            loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

            optimizer = loaded_graph.get_operation_by_name('adam')

            print('starting training ... ')
            for epoch in range(epochs):
                n_batches = 5
                for batch_i in range(1, n_batches + 1):
                        self._train_preprocess(sess,
                                            loaded_x, loaded_y, loaded_optimizer, loaded_acc,
                                            epoch, batch_i, batch_size, valid_set)

            # Save Model
            saver = tf.train.Saver()
            save_path = saver.save(sess, save_model_path)

    def _train_preprocess(self, sess,
                        input, label, optimizer, accuracy,
                        epoch, batch_i, batch_size, valid_set):
        tmpValidFeatures, valid_labels = valid_set

        for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
            _ = sess.run(optimizer,
                        feed_dict={input: batch_features,
                                   label: batch_labels})

        print('Epoch {:>2}, Preprocess Batch {}: '.format(epoch + 1, batch_i), end='')

        # calculate the mean accuracy over all validation dataset
        valid_acc = 0
        for batch_valid_features, batch_valid_labels in batch_features_labels(tmpValidFeatures, valid_labels, batch_size):
            valid_acc += sess.run(accuracy,
                                feed_dict={input:batch_valid_features,
                                           label:batch_valid_labels})

        tmp_num = tmpValidFeatures.shape[0]/batch_size
        print('Validation Accuracy {:.6f}'.format(valid_acc/tmp_num))

    def train(self, epochs, batch_size, valid_set, save_model_path):
        tmpValidFeatures, valid_labels = valid_set

        with tf.Session() as sess:
            print('global_variables_initializer...')
            sess.run(tf.global_variables_initializer())

            print('starting training ... ')
            for epoch in range(epochs):
                n_batches = 5
                for batch_i in range(1, n_batches + 1):
                        self._train_preprocess(sess,
                                            self.input, self.label, self.optimizer, self.accuracy,
                                            epoch, batch_i, batch_size, valid_set)

            # Save Model
            saver = tf.train.Saver()
            save_path = saver.save(sess, save_model_path)



def main ():
    dataset_path = './reduced dataset'
    learning_rate = 10
    epochs = 15
    batch_size = 5

    