# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
from numpy import arange
import tensorflow as tf
from skimage import io
from utils import dir_concat
from os.path import join
from tqdm import tqdm

use_DTM = True

tf.app.flags.DEFINE_string('gpu', '2', '')
tf.app.flags.DEFINE_string('HOME', '/media/data1/LanDCNN', '')
tf.app.flags.DEFINE_string('task_name', 'test', '')
tf.app.flags.DEFINE_string('phase', 'validation', '')
tf.app.flags.DEFINE_integer('num_of_channels', 7, '')
tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_float('expand_ratio', 0.25, '')
tf.app.flags.DEFINE_string('epoch', 'latest', '')

from validation_generator import data_padding
from augment_utils import norm

from nets import model
from utils import read_info_from_txt, write_txt

FLAGS = tf.app.flags.FLAGS
landslide_rgb = [255, 0, 0]
building_rgb = [255, 0, 255]

input_size = FLAGS.input_size
expand_ratio = FLAGS.expand_ratio
kernel_size = int(input_size * (1 - 2 * expand_ratio))

output_dir = join(FLAGS.HOME, 'eval', FLAGS.task_name)
os.system('rm -rf {}'.format(output_dir))
os.mkdir(output_dir)
data = np.load(join(FLAGS.HOME, 'dataset/preprocessed', '{}.npy'.format(FLAGS.phase)))
rows, cols = data.shape[:2]
predict_map = np.zeros((rows, cols))
ground_truth_map = np.zeros((rows, cols))
categorical_prob_map = np.zeros((rows, cols, 3))

def main(argv=None):
    assert len(FLAGS.gpu) == 1
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    with tf.get_default_graph().as_default():
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, FLAGS.num_of_channels], name='input_image')
        global_step = tf.get_variable('global_step', shape=[], initializer=tf.constant_initializer(0), trainable=False)

        logits = model.model(input_image, is_training=False)
        prob = tf.nn.softmax(logits)
        variables_average = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variables_average.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            checkpoint_path = join(FLAGS.HOME, 'checkpoint', FLAGS.task_name, 'model_{}.ckpt'.format(FLAGS.epoch))
            print('INFO: Inference using model: {}'.format(checkpoint_path))
            saver.restore(sess, checkpoint_path)

            for row_id in tqdm(arange(0, rows-1, kernel_size)):
                for col_id in arange(0, cols-1, kernel_size):
                    row_id, col_id = int(row_id), int(col_id)
                    # print(row_id, col_id)
                    tile = data_padding(data, row_id, col_id)
                    tile[..., :FLAGS.num_of_channels] = norm(tile[..., :FLAGS.num_of_channels])
                    output_prob = sess.run(prob, feed_dict={input_image: np.expand_dims(tile[:, :, :FLAGS.num_of_channels], axis=0)})
                    row_step_size, col_step_size = int(kernel_size), int(kernel_size)
                    if row_id + kernel_size > rows - 1:
                        row_step_size = int(rows - 1 - row_id)
                    if col_id + kernel_size > cols - 1:
                        col_step_size = int(cols - 1 - col_id)
                    start_margin = int(expand_ratio * input_size)
                    output_prob = output_prob[0, start_margin:start_margin + row_step_size, start_margin:start_margin + col_step_size, :]

                    category_map = np.argmax(output_prob, axis=-1)
                    prob_map = np.max(output_prob, axis=-1)
                    category_map[prob_map < 0.5] = 0
                    # scars_map = output_prob[0, start_margin:start_margin + row_step_size, start_margin:start_margin + col_step_size, 1] > 0.9

                    predict_map[row_id:row_id + row_step_size, col_id:col_id + col_step_size] = category_map
                    categorical_prob_map[row_id:row_id + row_step_size, col_id:col_id + col_step_size] = output_prob


                    ground_truth_map[row_id:row_id+row_step_size, col_id:col_id+col_step_size] = \
                        np.maximum(tile[start_margin:start_margin + row_step_size, start_margin:start_margin + col_step_size, -1], 0)

            io.imsave(join(output_dir, 'predictions_{}.jpg'.format(FLAGS.epoch)), predict_map)
            io.imsave(join(output_dir, 'ground_truth_{}.jpg'.format(FLAGS.epoch)), ground_truth_map)
            np.save(join(output_dir, 'prob.npy'), categorical_prob_map)

    # correct_pred = np.sum((predict_map[:, :] == ground_truth_map[:, :]) * ground_truth_map)
    # pred = np.sum(predict_map) + np.sum(ground_truth_map) - correct_pred
    # print('INFO: Overall accuracy on validation dataset is {:04f}'.format(correct_pred / pred))

if __name__ =='__main__':
    tf.app.run()


