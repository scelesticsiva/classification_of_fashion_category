"""
@Modified by - Sivaramakrishnan
taken from (https://github.com/machrisaa/tensorflow-vgg)
"""
import tensorflow as tf
import numpy as np

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg16:
    def __init__(self, devices,vgg16_npy_path=None):
        self.devices = devices
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()

    def build(self, rgb):
        """
        Build and loads the pre-trained vgg weights
        :param rgb: image in RGB format
        :return: fc-7 layer output for each image being given as input
        """
        rgb_scaled = rgb * 255.0
        bgr = rgb_scaled[..., ::-1]
        means = tf.stack([tf.constant(VGG_MEAN[0], shape=[224, 224]), tf.constant(VGG_MEAN[1], shape=[224, 224]),
                          tf.constant(VGG_MEAN[2], shape=[224, 224])], axis=2)
        bgr = tf.subtract(bgr,means)

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.data_dict = None

        return self.relu7

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            with tf.device(self.devices[0]):
                filt = self.get_conv_filter(name)
                conv_biases = self.get_bias(name)

            with tf.device(self.devices[1]):
                conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
                bias = tf.nn.bias_add(conv, conv_biases)
                relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            with tf.device(self.devices[0]):
                weights = self.get_fc_weight(name)
                biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            with tf.device(self.devices[1]):
                fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")