import tensorflow as tf

def weight(shape, initializer='truncated_normal', stddev=1e-1):
	'''
	Only truncated normal is implemented
	Implement Xavier
	'''

	initializer = tf.truncated_normal(shape, dtype=tf.float32, stddev=stddev)
	W = tf.Variable(initializer, name = 'W')
	return W

def bias(value, shape):
	initializer = tf.constant(value=value, shape=shape, dtype=tf.float32)
	b = tf.Variable(initializer, name = 'b')
	return b

def conv2d(input_layer, size_in, size_out, ksize=(3,3), strides=[1,1,1,1], padding='SAME', init = 'truncated_normal', stddev = 1e-1, name='conv'):
	'''
	Does a 2D convolution on the input
	'''
	with tf.name_scope(name) as scope:
		shape = [ksize[0], ksize[1], size_in, size_out]
		W = weight(shape, init, stddev)
		b = bias(0.0, [size_out])
		conv = tf.nn.conv2d(input_layer, W, strides, padding)
		out = tf.nn.bias_add(conv, b)
		act = tf.nn.relu(out)
	return W, b, act
	

def fully_connected(input_layer, size_in, size_out, init='truncated_normal', stddev=1e-1, name='fully_connected'):
	with tf.name_scope(name) as scope:
		shape = [size_in, size_out]
		W = weight(shape, init, stddev)
		b = bias(0.0, [size_out])
		out = tf.nn.bias_add(tf.matmul(input_layer, W), b)
		act = tf.nn.relu(out)
	return W, b, act


def max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool'):
	return tf.nn.max_pool(input, ksize, strides, padding, name='pool')