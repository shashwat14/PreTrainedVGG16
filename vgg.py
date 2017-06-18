"""
The weights have been used from the files made available by Davi Frossard
The code is changed heavily to make it more modular, cleaner and allow visualization
The weights are obtained from the following web page : https://www.cs.toronto.edu/~frossard/post/vgg16/
"""

from layers import *
import tensorflow as tf
import numpy as np

class VGG16():

	def __init__(self, sess, inputs, only_convolution=False):
		self.only_convolution = only_convolution
		self.paramters = []
		self.inputs = inputs
		self.sess = sess
		self.preprocess()
		self.convolution_layers()
		if not self.only_convolution:
			self.fully_connected_layer()
			self.belief()

	def preprocess(self):
		with tf.name_scope('preprocess') as scope:
			mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1,1,1,3], name='img_mean')
			self.preprocessed = self.inputs - mean
			return self.preprocessed

	def convolution_layers(self):
		'''
		Param: None
		Return: convolution codes after reshaping to [None,7*7*512]
		'''
		#Block 1
		W, b, conv1_1 = conv2d(self.preprocessed, 3, 64, name='conv1_1')
		self.paramters += [W,b]
		W, b, conv1_2 = conv2d(conv1_1, 64, 64, name='conv1_2')
		self.paramters += [W,b]
		pool_1 = max_pool(conv1_2, name='pool_1')

		#Block 2
		W, b, conv2_1 = conv2d(pool_1, 64, 128, name='conv2_1')
		self.paramters += [W,b]
		W, b, conv2_2 = conv2d(conv2_1, 128, 128, name='conv2_2')
		self.paramters += [W,b]
		pool_2 = max_pool(conv2_2, name='pool_2')

		#Block 3
		W, b, conv3_1 = conv2d(pool_2, 128, 256, name='conv3_1')
		self.paramters += [W,b]
		W, b, conv3_2 = conv2d(conv3_1, 256, 256, name='conv3_2')
		self.paramters += [W,b]
		W, b, conv3_3 = conv2d(conv3_2, 256, 256, name='conv3_3')
		self.paramters += [W,b]
		pool_3 = max_pool(conv3_3, name='pool_3')

		#Block 4
		W, b, conv4_1 = conv2d(pool_3, 256, 512, name='conv4_1')
		self.paramters += [W,b]
		W, b, conv4_2 = conv2d(conv4_1, 512, 512, name='conv4_2')
		self.paramters += [W,b]
		W, b, conv4_3 = conv2d(conv4_2, 512, 512, name='conv4_3')
		self.paramters += [W,b]
		pool_4 = max_pool(conv4_3, name='pool_4')

		#Block 5
		W, b, conv5_1 = conv2d(pool_4, 512, 512, name='conv5_1')
		self.paramters += [W,b]
		W, b, conv5_2 = conv2d(conv5_1, 512, 512, name='conv5_2')
		self.paramters += [W,b]
		W, b, conv5_3 = conv2d(conv5_2, 512, 512, name='conv5_3')
		self.paramters += [W,b]
		pool_5 = max_pool(conv5_3, name='pool_5')

		self.convolution_codes = tf.reshape(pool_5, [-1, 7*7*512])

	def fully_connected_layer(self):
		'''
		Build fully connected module
		Param: None
		Return: Logits
		'''
		W, b, fc1 = fully_connected(self.convolution_codes, 7*7*512, 4096)
		self.paramters += [W,b]
		W, b, fc2 = fully_connected(fc1, 4096, 4096)
		self.paramters += [W,b]
		W, b, fc3 = fully_connected(fc2, 4096, 1000)
		self.paramters += [W,b]
		self.logits = fc3

	def belief(self):
		'''
		Param: None
		Return: Probability scores
		'''
		self.probs = tf.nn.softmax(self.logits)

	def load_weights(self, path):
		
		weights = np.load(path)
		keys = sorted(weights.keys())
		for i, k in enumerate(keys):

			#Skip intializing fully_connected layers
			if i > 25 and self.only_convolution:
				continue

			#Load weights to layers
			print i, k, np.shape(weights[k])
			self.sess.run(self.paramters[i].assign(weights[k]))
	