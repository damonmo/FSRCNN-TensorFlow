import tensorflow as tf
from utils import tf_ssim

class Model(object):

  def __init__(self, config):
    self.name = "Waifu2x"
    self.model_params = [1, 16, 32, 64, 128, 256]
    self.scale = config.scale
    self.radius = config.radius
    self.padding = config.padding
    self.images = config.images
    self.batch = config.batch
    self.label_size = config.label_size

  def model(self):
    d = self.model_params
    m = len(d) + 2


    for i in range(len(self.model_params)-1):
        in_channels = self.model_params[i]
        out_channels = self.model_params[i+1]        
        weights = tf.get_variable('w_{}'.format(out_channels), shape=[3, 3, in_channels, out_channels], initializer=tf.variance_scaling_initializer(0.1))
        biases = tf.get_variable('b_{}'.format(out_channels), initializer=tf.zeros([out_channels]))
        if i == 0:
            conv = tf.nn.conv2d(self.images, weights, strides=[1,1,1,1], padding='VALID', data_format='NHWC')
        else:
            conv = tf.nn.conv2d(conv, weights, strides=[1,1,1,1], padding='VALID', data_format='NHWC')
        conv = tf.nn.bias_add(conv, biases, data_format='NHWC')
        conv = self.prelu(conv, 1)

    # Sub-pixel convolution
    size = self.radius * 2 + 1
    deconv_weights = tf.get_variable('deconv_w', shape=[size, size, self.model_params[-1], self.scale**2], initializer=tf.variance_scaling_initializer(0.01))
    deconv_biases = tf.get_variable('deconv_b', initializer=tf.zeros([self.scale**2]))
    deconv = tf.nn.conv2d(conv, deconv_weights, strides=[1,1,1,1], padding='SAME', data_format='NHWC')
    deconv = tf.nn.bias_add(deconv, deconv_biases, data_format='NHWC')
    deconv = tf.depth_to_space(deconv, self.scale, name='pixel_shuffle', data_format='NHWC')

    return deconv

  def prelu(self, _x, i):
    """
    PreLU tensorflow implementation
    """
    alphas = tf.get_variable('alpha{}'.format(i), _x.get_shape()[-1], initializer=tf.constant_initializer(0.2), dtype=tf.float32)

    return tf.nn.relu(_x) - alphas * tf.nn.relu(-_x)

  def loss(self, Y, X):
    dY = tf.image.sobel_edges(Y)
    dX = tf.image.sobel_edges(X)
    M = tf.sqrt(tf.square(dY[:,:,:,:,0]) + tf.square(dY[:,:,:,:,1]))
    return tf.losses.absolute_difference(dY, dX) \
         + tf.losses.absolute_difference((1.0 - M) * Y, (1.0 - M) * X, weights=2.0)
