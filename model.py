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
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size for training.""")

tf.app.flags.DEFINE_string('summary_dir', 'logs/',
                           """Path of where to store the summary files.""")
tf.app.flags.DEFINE_string('image_dir', 'images/',
                           """Path of where to store the summary files.""")


INPUT_CHANNELS = 3

# Generator Layers
# TODO: Make sure deconv filter size is right
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

# Discriminator Layers
CONV0_FILTER_SIZE = 2
CONV0_FILTER_STRIDE = 2
CONV0_NUM_FILTERS = 128

CONV1_FILTER_SIZE = 2
CONV1_FILTER_STRIDE = 2
CONV1_NUM_FILTERS = 256

CONV2_FILTER_SIZE = 2
CONV2_FILTER_STRIDE = 2
CONV2_NUM_FILTERS = 512


def leaky_relu(x, alpha, name):
    return tf.maximum(alpha * x, x, name=name)


# TODO: Add Batchnorm
def conv_layer(input_tensor, weight_init, filter_size, filter_stride, 
               num_filters, in_channels, nonlinear_func, name):
    weight_shape = [filter_size, filter_size, in_channels, num_filters]
    # conv_weights = np.random.normal(scale=weight_init, size=weight_shape)
    # conv_weights = tf.Variable(conv_weights, dtype=tf.float32,
    #                            name=name + '/weights')
    # bias_init = np.zeros(num_filters)
    # bias = tf.Variable(bias_init, dtype=tf.float32, name=name+ '/bias')
    initializer = tf.random_normal_initializer(stddev=weight_init)
    conv_weights = tf.get_variable(name + '/weights',
                                   shape=weight_shape,
                                   initializer=initializer)
    bias = tf.get_variable(name + '/bias',
                           shape=[num_filters],
                           initializer=tf.constant_initializer())

    stride = [1, filter_stride, filter_stride, 1]
    conv = tf.nn.conv2d(input_tensor, conv_weights, stride, padding='VALID',
                        name=name + '/affine')
    activation = nonlinear_func(tf.nn.bias_add(conv, bias), 
                                name=name + '/activation')
    return activation


# TODO: Add Batchnorm
def deconv_layer(input_tensor, weight_init, filter_size, filter_stride, 
                 num_filters, in_channels, output_size, nonlinear_func, name):
    weight_shape = [filter_size, filter_size, num_filters, in_channels]
    # deconv_weights = np.random.normal(scale=weight_init, size=weight_shape)
    # deconv_weights = tf.Variable(deconv_weights, dtype=tf.float32, 
    #                              name=name + '/weights')
    # bias_init = np.zeros(num_filters)
    # bias = tf.Variable(bias_init, dtype=tf.float32, name=name+ '/bias')
    initializer = tf.random_normal_initializer(stddev=weight_init)
    deconv_weights = tf.get_variable(name + '/weights',
                                     shape=weight_shape,
                                     initializer=initializer)
    bias = tf.get_variable(name + '/bias',
                           shape=[num_filters],
                           initializer=tf.constant_initializer())

    batch_size = tf.shape(input_tensor)[0]
    output_shape = tf.pack([batch_size, output_size, output_size, num_filters])
    stride = [1, filter_stride, filter_stride, 1]
    deconv = tf.nn.conv2d_transpose(input_tensor, deconv_weights, output_shape,
                                    stride, padding='SAME', 
                                    name=name + '/affine')
    activation = nonlinear_func(tf.nn.bias_add(deconv, bias), 
                                name=name + '/activation')
    return activation


def get_random_z(batch_size):
    return np.random.normal(size=[batch_size])


# TODO: add resize in decode tensor
def get_random_input_images(sess, image_dir, batch_size, 
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


def generator(input_tensor):
    z_in = tf.reshape(input_tensor, 
                      shape=[-1, FLAGS.z_size, FLAGS.z_size, 1])
    deconv0 = conv_layer(z_in, 
                         FLAGS.weight_init,
                         DECONV0_FILTER_SIZE,
                         DECONV0_FILTER_STRIDE,
                         DECONV0_NUM_FILTERS,
                         1,
                         tf.nn.relu,
                         'deconv0')
    deconv1 = deconv_layer(deconv0, 
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
                           'output')
    return gen_out


def discriminator(input_tensor):
    conv0 = conv_layer(input_tensor, 
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
    conv2 = conv_layer(conv1, 
                       FLAGS.weight_init,
                       CONV2_FILTER_SIZE,
                       CONV2_FILTER_STRIDE,
                       CONV2_NUM_FILTERS,
                       CONV1_NUM_FILTERS,
                       lambda x, name: leaky_relu(x, 
                                                  FLAGS.relu_slope, 
                                                  name),
                       'conv2')

    # Make the output a probability.
    num_parameters = CONV2_FILTER_SIZE * CONV2_FILTER_SIZE * \
            CONV2_NUM_FILTERS
    conv2_flatten = tf.reshape(conv2,
                               shape=[-1, num_parameters],
                               name='final_input')
    # weights = np.random.normal(scale=FLAGS.weight_init, 
    #                            size=[num_parameters, 1])
    # weights = tf.Variable(weights, dtype=np.float32, name='final_weights')
    # bias = np.zeros(size=[1])
    # bias = tf.Variable(bias, dtype=np.float32, name='final_bias')
    initializer = tf.random_normal_initializer(stddev=FLAGS.weight_init)
    weights = tf.get_variable('final_weights',
                              shape=[num_parameters, 1],
                              initializer=initializer)
    bias = tf.get_variable('final_bias',
                           shape=[1],
                           initializer=tf.constant_initializer())

    disc_out = tf.sigmoid(tf.matmul(conv2_flatten, weights) + bias,
                          name='output')
    return disc_out


def add_optimization(learning_rate, beta1, disc_gen, disc_true, 
                     gen_label, disc_label):
    gen_loss = tf.reduce_mean(tf.log(1 - disc_gen))
    disc_loss = -(tf.reduce_mean(tf.log(disc_true)) + gen_loss)
    
    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 scope=gen_label)
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  scope=disc_label)

    gen_opt = tf.AdamOptimizer(learning_rate=learning_rate,
                               beta1=beta1).minimize(gen_loss,
                                                     var_list=gen_vars)
    disc_opt = tf.AdamOptimizer(learning_rate=learning_rate,
                                beta1=beta1).minimize(gen_loss,
                                                      var_list=disc_vars)
    return gen_opt, disc_opt


def main(_):
    generator_label = 'Generator'
    discriminator_label = 'Discriminator'
    with tf.variable_scope(generator_label):
        z = tf.placeholder(tf.float32, 
                           shape=[None, int(FLAGS.z_size ** 2)], 
                           name='Z')
        gen_out = generator(z)

    with tf.variable_scope(discriminator_label) as scope:
        # x_gen = tf.placeholder(tf.float32,
        #                        shape=[None, 
        #                               DECONV4_OUT_SIZE, 
        #                               DECONV4_OUT_SIZE,
        #                               INPUT_CHANNELS],
        #                        name='X_gen')
        disc_gen = discriminator(gen_out)

        scope.reuse_variables()
        x_true = tf.placeholder(tf.float32,
                                shape=[None, 
                                       DECONV4_OUT_SIZE, 
                                       DECONV4_OUT_SIZE,
                                       INPUT_CHANNELS],
                                name='X_true')
        disc_true = discriminator(x_true)

    # Create the graph.
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    # Read image tensors.
    image_data_tensor = tf.placeholder(tf.string)
    decode_tensor = tf.image.decode_jpeg(image_data_tensor, 
                                         channels=INPUT_CHANNELS)

    # Add update steps
    gen_step, disc_step = add_optimization(FLAGS.learning_rate,
                                           FLAGS.beta1,
                                           disc_gen, 
                                           disc_true,
                                           generator_label, 
                                           discriminator_label)

    writer = tf.train.SummaryWriter(FLAGS.summary_dir, sess.graph)

    for step in xrange(FLAGS.num_steps):
        # Discriminator update step.
        batch_z = get_random_z(FLAGS.batch_size)
        batch_imgs = get_random_input_images(sess,
                                             FLAGS.image_dir,
                                             FLAGS.batch_size,
                                             image_data_tensor,
                                             decode_tensor)
        sess.run(disc_step, feed_dict={z: batch_z, x_true: batch_imgs})

        # Generator update step.
        batch_z = get_random_z(FLAGS.batch_size)
        sess.run(gen_step, feed_dict={z: batch_z})


if __name__ == '__main__':
    tf.app.run()

