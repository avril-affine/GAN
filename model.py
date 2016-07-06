import tensorflow as tf
import numpy as np
import os
import re

from tensorflow.python.platform import gfile
from datetime import datetime

FLAGS = tf.app.flags.FLAGS


# Optimization parameters.
tf.app.flags.DEFINE_integer('num_steps', 2000,
                            """How many training steps to run.""")
tf.app.flags.DEFINE_float('learning_rate', 0.0002,
                          """Learning rate suggested in paper.""")
tf.app.flags.DEFINE_float('beta1', 0.5,
                          """Beta1 suggested in paper.""")
tf.app.flags.DEFINE_float('beta2', 0.55,
                          """Beta2 for adam optimizer.""")
tf.app.flags.DEFINE_float('weight_init', 0.02,
                          """Weight initialization standard deviation.""")
tf.app.flags.DEFINE_float('relu_slope', 0.2,
                          """Slope to use for leaky ReLU.""")
tf.app.flags.DEFINE_integer('z_size', 100,
                            """Size of random noise input vector """
                            """for generator.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size for training.""")

tf.app.flags.DEFINE_string('summary_dir', 'logs/',
                           """Path of where to store the summary files.""")
tf.app.flags.DEFINE_string('image_dir', 'flickr_resize/',
                           """Path of where to store the summary files.""")


INPUT_CHANNELS = 3

# Generator Layers
# TODO: Make sure deconv filter size is right
# DECONV0_FILTER_SIZE = 7
# DECONV0_FILTER_STRIDE = 1
# DECONV0_NUM_FILTERS = 1024
DECONV0_FILTER_SIZE = 4
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
DECONV4_NUM_FILTERS = 16
DECONV4_OUT_SIZE = 64

DECONV5_FILTER_SIZE = 5     # Is this right filter size?
DECONV5_FILTER_STRIDE = 2
DECONV5_NUM_FILTERS = 3
DECONV5_OUT_SIZE = 128

# Discriminator Layers
CONV0_FILTER_SIZE = 5
CONV0_FILTER_STRIDE = 2
CONV0_NUM_FILTERS = 64

CONV1_FILTER_SIZE = 5
CONV1_FILTER_STRIDE = 2
CONV1_NUM_FILTERS = 128

CONV2_FILTER_SIZE = 5
CONV2_FILTER_STRIDE = 2
CONV2_NUM_FILTERS = 256

CONV3_FILTER_SIZE = 5
CONV3_FILTER_STRIDE = 2
CONV3_NUM_FILTERS = 512

CONV4_FILTER_SIZE = 5
CONV4_FILTER_STRIDE = 2
CONV4_NUM_FILTERS = 1024


def leaky_relu(x, alpha, name):
    return tf.maximum(alpha * x, x, name=name)


def batch_norm(x, n_out, phase_train, name='bn'):
    beta = tf.get_variable(name + '/beta',
                           shape=[n_out],
                           initializer=tf.constant_initializer())
    gamma = tf.get_variable(name + '/gamma',
                            shape=[n_out],
                            initializer=tf.random_normal_initializer(1., 0.02))
    # beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
    #                                name='beta', trainable=True)
    # gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
    #                                 name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], 
                                          name=name + '/moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), 
                                 ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
    if not tf.get_variable_scope().reuse:
        tf.histogram_summary('summary/beta/' + name, beta)
        tf.histogram_summary('summary/gamma/' + name, gamma)
        tf.histogram_summary('summary/normed/' + name, normed)
    return normed


def conv_layer(input_tensor, mode_tensor, weight_init, filter_size, 
               filter_stride, num_filters, in_channels, nonlinear_func, 
               use_batchnorm, name):
    # Init variables
    weight_shape = [filter_size, filter_size, in_channels, num_filters]
    initializer = tf.random_normal_initializer(stddev=weight_init)
    conv_weights = tf.get_variable(name + '/weights',
                                   shape=weight_shape,
                                   initializer=initializer)
    bias = tf.get_variable(name + '/bias',
                           shape=[num_filters],
                           initializer=tf.constant_initializer())

    # Apply convolution
    stride = [1, filter_stride, filter_stride, 1]
    conv = tf.nn.conv2d(input_tensor, conv_weights, stride, padding='SAME',
                        name=name + '/affine')
    # Apply batchnorm
    if use_batchnorm:
        conv = batch_norm(conv, num_filters, tf.equal(mode_tensor, 'train'),
                          name + '/bn')
            
    activation = nonlinear_func(tf.nn.bias_add(conv, bias), 
                                name=name + '/activation')

    if not tf.get_variable_scope().reuse:
        tf.histogram_summary('summary/weights/' + name, conv_weights)
        tf.histogram_summary('summary/activations/' + name, activation)
    return activation


def deconv_layer(input_tensor, mode_tensor, weight_init, filter_size, 
                 filter_stride, num_filters, in_channels, output_size, 
                 nonlinear_func, use_batchnorm, name):
    # Initialize variables
    weight_shape = [filter_size, filter_size, num_filters, in_channels]
    initializer = tf.random_normal_initializer(stddev=weight_init)
    deconv_weights = tf.get_variable(name + '/weights',
                                     shape=weight_shape,
                                     initializer=initializer)
    bias = tf.get_variable(name + '/bias',
                           shape=[num_filters],
                           initializer=tf.constant_initializer())

    # Apply deconvolution
    # batch_size = tf.shape(input_tensor)[0]
    # output_shape = tf.pack([batch_size, output_size, output_size, num_filters])
    # TODO: batchnorm needs last dimension shape but pack takes that away
    output_shape = [FLAGS.batch_size, output_size, output_size, num_filters]
    stride = [1, filter_stride, filter_stride, 1]
    deconv = tf.nn.conv2d_transpose(input_tensor, deconv_weights, output_shape,
                                    stride, padding='SAME', 
                                    name=name + '/affine')
    # Apply batchnorm
    if use_batchnorm:
        deconv = batch_norm(deconv, num_filters,
                            tf.equal(mode_tensor, 'train'),
                            name + '/bn')

    activation = nonlinear_func(tf.nn.bias_add(deconv, bias), 
                                name=name + '/activation')

    if not tf.get_variable_scope().reuse:
        tf.histogram_summary('summary/weights/' + name, deconv_weights)
        tf.histogram_summary('summary/activations/' + name, activation)
    return activation


def get_random_z(batch_size, z_size):
    return np.random.uniform(-1.0, 1.0, size=[batch_size, z_size])


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
        image = image * 2. / 255. - 1.
        images.append(image)
    return images


def generator(input_tensor, mode_tensor):
    num_features = DECONV0_FILTER_SIZE * DECONV0_FILTER_SIZE * \
            DECONV0_NUM_FILTERS
    initializer = tf.random_normal_initializer(stddev=FLAGS.weight_init)
    proj_weights = tf.get_variable('deconv0/weights',
                                   shape=[FLAGS.z_size, num_features],
                                   initializer=initializer)
    proj_bias = tf.get_variable('deconv0/bias',
                                shape=[num_features],
                                initializer=tf.constant_initializer())
    deconv0 = tf.matmul(input_tensor, proj_weights) + proj_bias
    deconv0 = tf.reshape(deconv0, [-1, 
                                   DECONV0_FILTER_SIZE,
                                   DECONV0_FILTER_SIZE,
                                   DECONV0_NUM_FILTERS])
    deconv0 = tf.nn.relu(deconv0, name='deconv0')
    # z_in = tf.reshape(input_tensor, 
    #                   shape=[-1, FLAGS.z_size, FLAGS.z_size, 1])
    # deconv0 = conv_layer(z_in, 
    #                      mode_tensor,
    #                      FLAGS.weight_init,
    #                      DECONV0_FILTER_SIZE,
    #                      DECONV0_FILTER_STRIDE,
    #                      DECONV0_NUM_FILTERS,
    #                      1,
    #                      tf.nn.relu,
    #                      True,
    #                      'deconv0')
    deconv1 = deconv_layer(deconv0, 
                           mode_tensor,
                           FLAGS.weight_init,
                           DECONV1_FILTER_SIZE,
                           DECONV1_FILTER_STRIDE,
                           DECONV1_NUM_FILTERS,
                           DECONV0_NUM_FILTERS,
                           DECONV1_OUT_SIZE,
                           tf.nn.relu,
                           True,
                           'deconv1')
    deconv2 = deconv_layer(deconv1, 
                           mode_tensor,
                           FLAGS.weight_init,
                           DECONV2_FILTER_SIZE,
                           DECONV2_FILTER_STRIDE,
                           DECONV2_NUM_FILTERS,
                           DECONV1_NUM_FILTERS,
                           DECONV2_OUT_SIZE,
                           tf.nn.relu,
                           True,
                           'deconv2')
    deconv3 = deconv_layer(deconv2, 
                           mode_tensor,
                           FLAGS.weight_init,
                           DECONV3_FILTER_SIZE,
                           DECONV3_FILTER_STRIDE,
                           DECONV3_NUM_FILTERS,
                           DECONV2_NUM_FILTERS,
                           DECONV3_OUT_SIZE,
                           tf.nn.relu,
                           True,
                           'deconv3')
    deconv4 = deconv_layer(deconv3, 
                           mode_tensor,
                           FLAGS.weight_init,
                           DECONV4_FILTER_SIZE,
                           DECONV4_FILTER_STRIDE,
                           DECONV4_NUM_FILTERS,
                           DECONV3_NUM_FILTERS,
                           DECONV4_OUT_SIZE,
                           tf.nn.relu,
                           True,
                           'deconv4')
    gen_out = deconv_layer(deconv4, 
                           mode_tensor,
                           FLAGS.weight_init,
                           DECONV5_FILTER_SIZE,
                           DECONV5_FILTER_STRIDE,
                           DECONV5_NUM_FILTERS,
                           DECONV4_NUM_FILTERS,
                           DECONV5_OUT_SIZE,
                           tf.tanh,
                           True,
                           'gen_out')
    return gen_out


def discriminator(input_tensor, mode_tensor):
    conv0 = conv_layer(input_tensor, 
                       mode_tensor,
                       FLAGS.weight_init,
                       CONV0_FILTER_SIZE,
                       CONV0_FILTER_STRIDE,
                       CONV0_NUM_FILTERS,
                       INPUT_CHANNELS,
                       lambda x, name: leaky_relu(x, 
                                                  FLAGS.relu_slope, 
                                                  name),
                       False,
                       'conv0')
    conv1 = conv_layer(conv0, 
                       mode_tensor,
                       FLAGS.weight_init,
                       CONV1_FILTER_SIZE,
                       CONV1_FILTER_STRIDE,
                       CONV1_NUM_FILTERS,
                       CONV0_NUM_FILTERS,
                       lambda x, name: leaky_relu(x, 
                                                  FLAGS.relu_slope, 
                                                  name),
                       True,
                       'conv1')
    conv2 = conv_layer(conv1, 
                       mode_tensor,
                       FLAGS.weight_init,
                       CONV2_FILTER_SIZE,
                       CONV2_FILTER_STRIDE,
                       CONV2_NUM_FILTERS,
                       CONV1_NUM_FILTERS,
                       lambda x, name: leaky_relu(x, 
                                                  FLAGS.relu_slope, 
                                                  name),
                       True,
                       'conv2')
    conv3 = conv_layer(conv2, 
                       mode_tensor,
                       FLAGS.weight_init,
                       CONV3_FILTER_SIZE,
                       CONV3_FILTER_STRIDE,
                       CONV3_NUM_FILTERS,
                       CONV2_NUM_FILTERS,
                       lambda x, name: leaky_relu(x, 
                                                  FLAGS.relu_slope, 
                                                  name),
                       True,
                       'conv3')
    conv4 = conv_layer(conv3, 
                       mode_tensor,
                       FLAGS.weight_init,
                       CONV4_FILTER_SIZE,
                       CONV4_FILTER_STRIDE,
                       CONV4_NUM_FILTERS,
                       CONV3_NUM_FILTERS,
                       lambda x, name: leaky_relu(x, 
                                                  FLAGS.relu_slope, 
                                                  name),
                       True,
                       'conv4')

    # Make the output a probability.
    # num_parameters = CONV4_FILTER_SIZE * CONV4_FILTER_SIZE * \
    #         CONV4_NUM_FILTERS
    conv4_flatten = tf.reshape(conv4,
                               shape=[FLAGS.batch_size, -1],
                               name='final_input')
    # weights = np.random.normal(scale=FLAGS.weight_init, 
    #                            size=[num_parameters, 1])
    # weights = tf.Variable(weights, dtype=np.float32, name='final_weights')
    # bias = np.zeros(size=[1])
    # bias = tf.Variable(bias, dtype=np.float32, name='final_bias')
    initializer = tf.random_normal_initializer(stddev=FLAGS.weight_init)
    # weights = tf.get_variable('final_weights',
    #                           shape=[num_parameters, 1],
    #                           initializer=initializer)
    weights = tf.get_variable('final_weights',
                              shape=[conv4_flatten.get_shape()[-1], 1],
                              initializer=initializer)
    bias = tf.get_variable('final_bias',
                           shape=[1],
                           initializer=tf.constant_initializer())

    # disc_out = tf.sigmoid(tf.matmul(conv4_flatten, weights) + bias,
    #                       name='output')
    disc_out = tf.add(tf.matmul(conv4_flatten, weights), bias,
                      name='disc_out')
    return disc_out


def add_optimization(learning_rate, beta1, beta2, disc_gen, disc_true, 
                     gen_label, disc_label):
    # gen_loss = tf.reduce_mean(-tf.log(1 - disc_gen), name='gen_loss')
    # disc_loss = tf.sub(tf.reduce_mean(tf.log(disc_true)), gen_loss,
    #                    name='disc_loss')
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_gen, tf.ones_like(disc_gen)), name='gen_loss')

    disc_g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_gen, tf.zeros_like(disc_gen)))
    disc_x_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_true, tf.ones_like(disc_true)))
    disc_loss = tf.add(disc_g_loss, disc_x_loss, name='disc_loss')
    
    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 scope=gen_label)
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  scope=disc_label)
    # print 'gen vars---------------------'
    # for v in gen_vars:
    #     print v.name
    # print 'disc vars----------------'
    # for v in disc_vars:
    #     print v.name

    gen_opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                     beta1=beta1,
                                     beta2=beta2).minimize(gen_loss,
                                                           var_list=gen_vars)
    disc_opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                      beta1=beta1,
                                      beta2=beta2).minimize(disc_loss,
                                                            var_list=disc_vars)
    return gen_loss, disc_loss, gen_opt, disc_opt


def main(_):

    generator_label = 'Generator'
    discriminator_label = 'Discriminator'
    mode_tensor = tf.placeholder(tf.string, shape=[], name='mode')
    with tf.variable_scope(generator_label):
        z = tf.placeholder(tf.float32, 
                           shape=[None, FLAGS.z_size], 
                           name='Z')
        gen_out = generator(z, mode_tensor)

    with tf.variable_scope(discriminator_label) as scope:
        disc_gen = discriminator(gen_out, mode_tensor)

        scope.reuse_variables()
        x_true = tf.placeholder(tf.float32,
                                shape=[None, 
                                       DECONV5_OUT_SIZE, 
                                       DECONV5_OUT_SIZE,
                                       INPUT_CHANNELS],
                                name='X_true')
        disc_true = discriminator(x_true, mode_tensor)

    # Read image tensors.
    image_data_tensor = tf.placeholder(tf.string)
    decode_tensor = tf.image.decode_jpeg(image_data_tensor, 
                                         channels=INPUT_CHANNELS)

    # Add update steps
    gen_loss, disc_loss, gen_step, disc_step = (
            add_optimization(FLAGS.learning_rate,
                             FLAGS.beta1,
                             FLAGS.beta2,
                             disc_gen, 
                             disc_true,
                             generator_label, 
                             discriminator_label))

    # Create graph
    sess = tf.Session()
    saver = tf.train.Saver()

    checkpoint_file = os.path.join(FLAGS.summary_dir, 'checkpoint')
    step_0 = 0
    if os.path.exists(checkpoint_file):
        print 'Restoring checkpoint'
        with open(checkpoint_file, 'r') as f:
            line = f.readline().strip()
        model_ckpt = line.split(': ')[1]
        model_ckpt = model_ckpt.strip('"')
        step_0 = int(model_ckpt.split('-')[-1])
        model_ckpt = os.path.join(FLAGS.summary_dir, model_ckpt)
        saver.restore(sess, model_ckpt)
    else:
        init = tf.initialize_all_variables()
        sess.run(init)

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(FLAGS.summary_dir, sess.graph)
    gen_to_img = tf.cast((gen_out + 1) * 255 / 2,
                         tf.uint8,
                         name='generated_image')
    img_summary = tf.image_summary('Generator Images', gen_to_img)
    gen_loss_summ = tf.scalar_summary('Generator Loss', gen_loss)
    disc_loss_summ = tf.scalar_summary('Discriminator Loss', disc_loss)

    n_train = len([f for f in os.listdir(FLAGS.image_dir)
                   if not f.startswith('.')])
    n_epoch = n_train / FLAGS.batch_size

    for step in xrange(step_0 + 1, FLAGS.num_steps):
        # Discriminator update step.
        batch_z = get_random_z(FLAGS.batch_size, FLAGS.z_size)
        batch_imgs = get_random_input_images(sess,
                                             FLAGS.image_dir,
                                             FLAGS.batch_size,
                                             image_data_tensor,
                                             decode_tensor)
        results = sess.run([disc_loss, disc_loss_summ, merged, disc_step], 
                           feed_dict={z: batch_z, 
                                      x_true: batch_imgs, 
                                      mode_tensor: 'train'})
        step_loss, disc_loss_str, merged_str, _ = results
        print '{} | Step {} | Loss = {}'.format(datetime.now(), 
                                                step, 
                                                step_loss)

        # Generator update step.
        # batch_z = get_random_z(FLAGS.batch_size, FLAGS.z_size))
        gen_loss_str, _ = sess.run([gen_loss_summ, gen_step],
                                   feed_dict={z: batch_z,
                                              mode_tensor: 'train'})
        sess.run(gen_step,
                 feed_dict={z: batch_z, mode_tensor: 'train'})

        writer.add_summary(merged_str, step)
        writer.add_summary(disc_loss_str, step)
        writer.add_summary(gen_loss_str, step)

        if step % n_epoch == 0 or step + 1 == FLAGS.num_steps:
            print 'Writing test image'
            img_str = sess.run(img_summary,
                               feed_dict={z: batch_z,
                                          mode_tensor: 'test'})
            writer.add_summary(img_str, step)
            model_file = os.path.join(FLAGS.summary_dir, 'model.ckpt')
            saver.save(sess, model_file, global_step=step)

    wrtier.close()


if __name__ == '__main__':
    tf.app.run()

