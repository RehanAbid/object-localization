import sys
import os.path
import tensorflow.compat.v1 as tf
import train_util as tu
from library import model
import numpy as np

def classify(
		image, 
		top_k, 
		k_patches, 
		ckpt_path, 
		imagenet_path):
	
	tf.disable_v2_behavior()
	wnids, words = tu.load_imagenet_meta(os.path.join(imagenet_path, 'data/meta.mat'))

	# taking a few crops from an image
	image_patches = tu.read_k_patches(image, k_patches)

	x = tf.placeholder(tf.float32, [None, 124, 124, 3])

	_, pred = model.classifier(x, dropout=1.0) 

	# calculate the average precision through the crops
	avg_prediction = tf.div(tf.reduce_sum(pred, 0), k_patches)

	# retrieve top 5 scores
	scores, indexes = tf.nn.top_k(avg_prediction, k=top_k)

	saver = tf.train.Saver()

	with tf.Session(config=tf.ConfigProto()) as sess:
		saver.restore(sess, os.path.join(ckpt_path, 'alexnet-cnn.ckpt'))

		s, i = sess.run([scores, indexes], feed_dict={x: image_patches})
		s, i = np.squeeze(s), np.squeeze(i)

		local = image_patches[1]
		print ('Location:\n{}'.format(local))
		print('Classification Results:')
		for idx in range(top_k):
			print ('{} - score: {}'.format(words[i[idx]], s[idx]))


if __name__ == '__main__':
	TOP_K = 5
	K_CROPS = 5
	IMAGENET_PATH = 'ILSVRC_Data/CLS-LOC'
	CKPT_PATH = 'ckpt-alexnet'

	image_path = sys.argv[1]

	classify(
		image_path, 
		TOP_K, 
		K_CROPS, 
		CKPT_PATH, 
		IMAGENET_PATH)

