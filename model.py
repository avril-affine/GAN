import tensorflow as tf
import numpy as np

from tensorflow.python.platform import gfile

FLAGS = tf.app.flags.FLAGS


# Optimization parameters.
tf.app.flags.DEFINE_integer('num_steps', 10000,
                            """How many training steps to run.""")
tf.app.flags.DEFINE_float('learning_rate', 0.0002,
                          """Learning rate suggested in paper.""")
tf.app.flags.DEFINE_float('beta1', 0.5,
                          """Beta1 suggested in paper.""")
tf.app.flags.DEFINE_float('weight_init', 0.02,
                          """Weight initialization standard deviation.""")
tf.app.flags.DEFINE_float('relu_slope', 0.2,
                          """Slope to use for leaky ReLU.""")
tf.app.flags.DEFINE_integer('z_size', 10,
                            """Square root size for input vector """
                            """for generator.""")


def leaky_relu(x, alpha, name):
    return tf.maximum(alpha * x, x, name=name)


# TODO: Add Batchnorm
def conv_layer(input_tensor, weight_init, filter_size, filter_stride, 
               num_filters, in_channels, nonlinear_func, name):
    size = [filter_size, filter_size, in_channels, num_filters]
    filter_init = np.random.normal(scale=weight_init, size=size)
    conv_filter = tf.Variable(filter_init, name=name + '_filter')
    stride = [1, filter_stride, filter_stride, 1]
    conv = tf.nn.conv2d(input_tensor, conv_filter, stride, padding='VALID'
                        name=name)
    bias_init = np.zeros(num_filters)
    bias = tf.Variable(bias_init, name=name+ '_bias')
    activation = nonlinear_func(conv + bias, name=name + '_activation')
    return activation


# TODO: Add Batchnorm
def deconv_layer(input_tensor, weight_init, filter_size, filter_stride, 
                 num_filters, in_channels, output_size, nonlinear_func, name):
    size = [filter_size, filter_size, num_filters, in_channels]
    filter_init = np.random.normal(scale=weight_init, size=size)
    deconv_filter = tf.Variable(filter_init, name=name + '_filter')
    output_shape = [None, output_size, output_size, num_filters]
    stride = [1, filter_stride, filter_stride, 1]
    deconv = tf.nn.conv2d_transpose(input_tensor, deconv_filter, output_shape,
                                    stride, padding='SAME', name=name)
    bias_init = np.zeros(num_filters)
    bias = tf.Variable(bias_init, name=name+ '_bias')
    activation = nonlinear_func(deconv + bias, name=name + '_activation')
    return activation


def random_generator_output(sess, z_tensor, output_tensor, batch_size, 
                            z_size, z=None):
    if not z:
        z = np.random.normal(size=[batch_size, int(z_size ** 2)])
    return sess.run(output_tensor, feed_dict={z_tensor: z})


# TODO: add resize in decode tensor
def random_input_images(sess, image_dir, batch_size, 
                        image_data_tensor, decode_tensor):
    filenames = [f for f in os.listdir(image_dir) if not f.startswith('.')]
    images = []
    for _ in xrange(batch_size):
        index = np.random.randint(0, len(filenames))
        image_path = os.path.join(image_dir, filenames[index])
        image_data = gfile.FastGFile(image_path, 'rb').read()
        image = sess.run(decode_tensor, 
                         feed_dict={image_data_tensor: image_data})
        images.append(image)
    return images


def discriminator_output(sess, input_images, x_tensor, output_tensor):
    return sess.run(output_tensor, feed_dict(x_tensor: input_images))
        

INPUT_CHANNELS = 3

# Generator layers
DECONV0_FILTER_SIZE = 7
DECONV0_FILTER_STRIDE = 1
DECONV0_NUM_FILTERS = 1024

DECONV1_FILTER_SIZE = 5     # Is this right filter size?
DECONV1_FILTER_STRIDE = 2
DECONV1_NUM_FILTERS = 512
DECONV1_OUT_SIZE = 8

DECONV2_FILTER_SIZE = 5     # Is this right filter size?
DECONV2_FILTER_STRIDE = 2
DECONV2_NUM_FILTERS = 256
DECONV2_OUT_SIZE = 16

DECONV3_FILTER_SIZE = 5     # Is this right filter size?
DECONV3_FILTER_STRIDE = 2
DECONV3_NUM_FILTERS = 128
DECONV3_OUT_SIZE = 32

DECONV4_FILTER_SIZE = 5     # Is this right filter size?
DECONV4_FILTER_STRIDE = 2
DECONV4_NUM_FILTERS = 3
DECONV4_OUT_SIZE = 64

# TODO: Check filter size and padding to make sure it outputs what you think.
# Discriminator layers
CONV0_FILTER_SIZE = 3       # Is this right filter size?
CONV0_FILTER_STRIDE = 2
CONV0_NUM_FILTERS = 128

CONV1_FILTER_SIZE = 3       # Is this right filter size?
CONV1_FILTER_STRIDE = 2
CONV1_NUM_FILTERS = 256

CONV2_FILTER_SIZE = 3       # Is this right filter size?
CONV2_FILTER_STRIDE = 2
CONV2_NUM_FILTERS = 512


def main(_):
    with tf.name_scope('Generator'):
        z = tf.placeholder(tf.float32, 
                           shape=[None, int(FLAGS.z_size ** 2)], 
                           name='Z')
        z_in = tf.reshape(z, shape=[None, FLAGS.z_size, FLAGS.z_size, 1])
        deconv0 = conv_layer(z_in, 
                             FLAGS.weight_init,
                             DECONV0_FILTER_SIZE,
                             DECONV0_FILTER_STRIDE,
                             DECONV0_NUM_FILTERS,
                             INPUT_CHANNELS,
                             tf.nn.relu,
                             'deconv0')
        deconv1 = deconv_layer(conv1, 
                               FLAGS.weight_init,
                               DECONV1_FILTER_SIZE,
                               DECONV1_FILTER_STRIDE,
                               DECONV1_NUM_FILTERS,
                               DECONV0_NUM_FILTERS,
                               DECONV1_OUT_SIZE,
                               tf.nn.relu,
                               'deconv1')
        deconv2 = deconv_layer(deconv1, 
                               FLAGS.weight_init,
                               DECONV2_FILTER_SIZE,
                               DECONV2_FILTER_STRIDE,
                               DECONV2_NUM_FILTERS,
                               DECONV1_NUM_FILTERS,
                               DECONV2_OUT_SIZE,
                               tf.nn.relu,
                               'deconv2')
        deconv3 = deconv_layer(deconv2, 
                               FLAGS.weight_init,
                               DECONV3_FILTER_SIZE,
                               DECONV3_FILTER_STRIDE,
                               DECONV3_NUM_FILTERS,
                               DECONV2_NUM_FILTERS,
                               DECONV3_OUT_SIZE,
                               tf.nn.relu,
                               'deconv3')
        gen_out = deconv_layer(deconv3, 
                               FLAGS.weight_init,
                               DECONV4_FILTER_SIZE,
                               DECONV4_FILTER_STRIDE,
                               DECONV4_NUM_FILTERS,
                               DECONV3_NUM_FILTERS,
                               DECONV4_OUT_SIZE,
                               tf.tanh,
                               'gen_out')

    with tf.name_scope('Discriminator'):
        x = tf.placeholder(tf.float32,
                           shape=[None, 
                                  DECONV4_OUT_SIZE, 
                                  DECONV4_OUT_SIZE,
                                  INPUT_CHANNELS],
                           name='X')
        conv0 = conv_layer(x, 
                           FLAGS.weight_init,
                           CONV0_FILTER_SIZE,
                           CONV0_FILTER_STRIDE,
                           CONV0_NUM_FILTERS,
                           INPUT_CHANNELS,
                           lambda x, name: leaky_relu(x, 
                                                      FLAGS.relu_slope, 
                                                      name),
                           'conv0')
        conv1 = conv_layer(conv0, 
                           FLAGS.weight_init,
                           CONV1_FILTER_SIZE,
                           CONV1_FILTER_STRIDE,
                           CONV1_NUM_FILTERS,
                           CONV0_NUM_FILTERS,
                           lambda x, name: leaky_relu(x, 
                                                      FLAGS.relu_slope, 
                                                      name),
                           'conv1')
        disc_out = conv_layer(x, 
                              FLAGS.weight_init,
                              CONV2_FILTER_SIZE,
                              CONV2_FILTER_STRIDE,
                              CONV2_NUM_FILTERS,
                              CONV1_NUM_FILTERS,
                              lambda x, name: leaky_relu(x, 
                                                         FLAGS.relu_slope, 
                                                         name),
                              'disc_out')
                           


if __name__ == '__main__':
    tf.app.run()

