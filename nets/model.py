import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

from nets import resnet_v1

FLAGS = tf.app.flags.FLAGS


def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2, tf.shape(inputs)[2]*2])


def batch_stardardize(images, is_training=False):
    print('INFO: Using batch standardize strategy.')
    batch_num = FLAGS.batch_size_per_gpu if is_training else 1
    print('INFO: Batch size per GPU is {}'.format(batch_num))
    channels = tf.split(axis=3, num_or_size_splits=FLAGS.num_of_channels, value=images)
    means, vars = tf.nn.moments(images, axes=[0,1,2])
    # assert len(means) == FLAGS.num_of_channels
    for i in range(FLAGS.num_of_channels):
        channels[i] -= means
        channels[i] /= vars + tf.constant(1e-5, dtype=tf.float32)
    return tf.concat(axis=3, values=channels)


def instance_stardardize(images, is_training=False):
    print('INFO: Using instance standardize strategy.')
    batch_num = FLAGS.batch_size_per_gpu if is_training else 1
    print('INFO: Batch size per GPU is {}'.format(batch_num))
    images_split = tf.split(axis=0, num_or_size_splits=batch_num, value=images)
    for i in range(batch_num):
        channels = tf.split(axis=3, num_or_size_splits=FLAGS.num_of_channels, value=images_split[i])
        mean, var = tf.nn.moments(images, axes=[0, 1, 2])
        for j in range(FLAGS.num_of_channels):
            channels[j] -= mean
            channels[j] /= var + tf.constant(1e-6, dtype=tf.float32)
        images_split[i] = tf.concat(axis=3, values=channels)
    return tf.concat(axis=0, values=images_split)


def model(images, weight_decay=1e-5, is_training=True):
    # if is_training:
    # images = instance_stardardize(images, is_training)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool6'], end_points['pool5'],
                 end_points['pool4'], end_points['pool3']]
            for i in range(4):
                print('INFO: Shape of resnet_block_outputs_{} (f): {}'.format(i, f[i].shape))
            g = [None, None, None, None]
            h = [None, None, None, None]
            filter_number = [None, 256, 128, 64]

            for i in range(4):
                if i == 0:
                    h[i] = f[i]
                else:
                    concat_and_merge_layer = slim.conv2d(tf.concat([g[i - 1], f[i]], axis=-1),
                                                         num_outputs=filter_number[i],
                                                         kernel_size=1)
                    h[i] = slim.conv2d(concat_and_merge_layer,
                                       num_outputs=filter_number[i],
                                       kernel_size=3)
                if i == 0:
                    g[i] = h[i]
                else:
                    g[i] = unpool(h[i])


                print('INFO: Shape of unpool_{} (g): {}'.format(i, g[i].shape))
                print('INFO: Shape of concat_and_merge_{} (h): {}'.format(i, h[i].shape))

            x = slim.conv2d(g[3], num_outputs=16, kernel_size=3)
            x = tf.image.resize_bilinear(x, size=[tf.shape(images)[1], tf.shape(images)[2]])

            softmax_logits = slim.conv2d(x, num_outputs=FLAGS.num_of_labels, kernel_size=1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)


            print('INFO: Shape of final score map: {}'.format(logits.shape))

            return softmax_logits


