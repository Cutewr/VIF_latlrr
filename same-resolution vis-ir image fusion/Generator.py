import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
# from Deconv import deconv_vis, deconv_ir

# 权重标准差
WEIGHT_INIT_STDDEV = 0.05


class Generator(object):

	# Generator包含两部分：编码器和解码器
	def __init__(self, sco):
		self.encoder = Encoder(sco)
		self.decoder = Decoder(sco)

	def transform(self, vis, ir):
		# 将红外和可见光在通道维度拼接在一起
		img = tf.concat([vis, ir], 3)
		# 先通过编码器
		code = self.encoder.encode(img)
		self.target_features = code
		# 再通过解码器生成最终的图像
		generated_img = self.decoder.decode(self.target_features)
		return generated_img


# 编码器负责从输入图像中提取特征
class Encoder(object):
	def __init__(self, scope_name):
		self.scope = scope_name
		self.weight_vars = []	# 存储每一层的权重
		with tf.variable_scope(self.scope):
			with tf.variable_scope('encoder'):
				self.weight_vars.append(self._create_variables(2, 48, 3, scope = 'conv1_1'))
				self.weight_vars.append(self._create_variables(48, 48, 3, scope = 'dense_block_conv1'))
				self.weight_vars.append(self._create_variables(96, 48, 3, scope = 'dense_block_conv2'))
				self.weight_vars.append(self._create_variables(144, 48, 3, scope = 'dense_block_conv3'))
				self.weight_vars.append(self._create_variables(192, 48, 3, scope = 'dense_block_conv4'))

	# self.weight_vars.append(self._create_variables(80, 32, 3, scope = 'dense_block_conv5'))

	# self.weight_vars.append(self._create_variables(96, 16, 3, scope = 'dense_block_conv6'))

	# 初始化每一层的卷积核和偏置
	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			# 卷积核采用截断正太分布初始化
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV),
			                     name = 'kernel')
			# 偏置初始化为0
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)

	def encode(self, image):
		dense_indices = [1, 2, 3, 4, 5]

		out = image
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]
			# 部分卷积操作会使用密集连接，即输入和输出拼在一起
			if i in dense_indices:
				out = conv2d(out, kernel, bias, dense = True, use_relu = True,
				             Scope = self.scope + '/encoder/b' + str(i))
			else:
				out = conv2d(out, kernel, bias, dense = False, use_relu = True,
				             Scope = self.scope + '/encoder/b' + str(i))
		return out


class Decoder(object):
	def __init__(self, scope_name):
		self.weight_vars = []
		self.scope = scope_name
		with tf.name_scope(scope_name):
			with tf.variable_scope('decoder'):
				self.weight_vars.append(self._create_variables(240, 240, 3, scope = 'conv2_1'))
				self.weight_vars.append(self._create_variables(240, 128, 3, scope = 'conv2_1'))
				self.weight_vars.append(self._create_variables(128, 64, 3, scope = 'conv2_2'))
				self.weight_vars.append(self._create_variables(64, 32, 3, scope = 'conv2_3'))
				self.weight_vars.append(self._create_variables(32, 1, 3, scope = 'conv2_4'))

	# self.weight_vars.append(self._create_variables(16, 1, 3, scope = 'conv2_5'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		with tf.variable_scope(scope):
			shape = [kernel_size, kernel_size, input_filters, output_filters]
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)

	def decode(self, image):
		final_layer_idx = len(self.weight_vars) - 1

		out = image
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]
			if i == 0:
				out = conv2d(out, kernel, bias, dense = False, use_relu = True,
				             Scope = self.scope + '/decoder/b' + str(i), BN = False)
			if i == final_layer_idx:
				out = conv2d(out, kernel, bias, dense = False, use_relu = False,
				             Scope = self.scope + '/decoder/b' + str(i), BN = False)
				out = tf.nn.tanh(out) / 2 + 0.5
			else:
				out = conv2d(out, kernel, bias, dense = False, use_relu = True, BN = True,
				             Scope = self.scope + '/decoder/b' + str(i))
		return out


def conv2d(x, kernel, bias, dense = False, use_relu = True, Scope = None, BN = True):
	# padding image with reflection mode 填充
	x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
	# conv and add bias 卷积+偏置
	out = tf.nn.conv2d(x_padded, kernel, strides = [1, 1, 1, 1], padding = 'VALID')
	out = tf.nn.bias_add(out, bias)
	# 批归一化
	if BN:
		with tf.variable_scope(Scope):
			out = tf.layers.batch_normalization(out, training = True)
	# 激活函数
	if use_relu:
		out = tf.nn.relu(out)
	# 如果dense为true，使用密集连接，将输入图像和卷积输出拼接在一起
	if dense:
		out = tf.concat([out, x], 3)
	return out

def up_sample(x, scale_factor = 2):
	_, h, w, _ = x.get_shape().as_list()
	new_size = [h * scale_factor, w * scale_factor]
	return tf.image.resize_nearest_neighbor(x, size = new_size)