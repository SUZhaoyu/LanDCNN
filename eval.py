# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
from numpy import arange
import tensorflow as tf
from skimage import io
from utils import dir_concat

from tqdm import tqdm

use_DTM = True

tf.app.flags.DEFINE_string('gpu', '2', '')
tf.app.flags.DEFINE_string('HOME', '/media/data1/ENTLI','')
tf.app.flags.DEFINE_string('task_name', 'weighted_entropy_10','')
tf.app.flags.DEFINE_string('dataset_name', 'test','')
tf.app.flags.DEFINE_string('phase', 'valid', '')
tf.app.flags.DEFINE_boolean('DTM', use_DTM, '')
if use_DTM:
    tf.app.flags.DEFINE_integer('num_of_channels', 7, '')
else:
    tf.app.flags.DEFINE_integer('num_of_channels', 6, '')
tf.app.flags.DEFINE_integer('input_size', 512, '')
# The idea was copied from original U-Net paper: https://arxiv.org/abs/1505.04597
tf.app.flags.DEFINE_float('expand_ratio', 0.25, '')
tf.app.flags.DEFINE_integer('batch_size', 15, '')
tf.app.flags.DEFINE_integer('num_of_labels', 2, '')
tf.app.flags.DEFINE_string('epoch', 'latest', '')
tf.app.flags.DEFINE_string('loss', 'weighted_entropy', '')
tf.app.flags.DEFINE_float('thres', 0.95, '')



from validation_generator import data_padding

from nets import model
from utils import read_info_from_txt, write_txt

FLAGS = tf.app.flags.FLAGS

input_size = FLAGS.input_size
expand_ratio = FLAGS.expand_ratio
kernel_size = int(input_size * (1 - 2 * expand_ratio))

output_dir = dir_concat(FLAGS.HOME, ['output', FLAGS.task_name])
print('INFO: Loading dataset from {}.'.format(dir_concat(FLAGS.HOME, ['dataset', FLAGS.dataset_name, '{}.npy'.format(FLAGS.phase)])))
data = np.load(dir_concat(FLAGS.HOME, ['dataset', FLAGS.dataset_name, '{}.npy'.format(FLAGS.phase)]))
rows, cols = data.shape[:2]
predict_map = np.zeros((rows, cols))
ground_truth_map = np.zeros((rows, cols))
means, stds = read_info_from_txt(dir_concat(FLAGS.HOME, ['dataset', FLAGS.dataset_name, '{}.txt'.format(FLAGS.phase)]))

for i in range(FLAGS.num_of_channels):
    data[:, :, i] -= means[i]
    data[:, :, i] /= stds[i]

def main(argv=None):
    assert len(FLAGS.gpu) == 1
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MkDir(output_dir)
    else:
        tf.gfile.DeleteRecursively(output_dir)
        tf.gfile.MkDir(output_dir)

    with tf.get_default_graph().as_default():
        pbar = tqdm(len(arange(0, rows - 1, kernel_size)) * len(arange(0, cols - 1, kernel_size)))
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, FLAGS.num_of_channels], name='input_image')
        global_step = tf.get_variable('global_step', shape=[], initializer=tf.constant_initializer(0), trainable=False)

        return_map = model.model(input_image, is_training=False)
        variables_average = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variables_average.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            checkpoint_path = dir_concat(FLAGS.HOME, ['checkpoint', FLAGS.task_name, 'model_{}.ckpt'.format(FLAGS.epoch)])
            print('INFO: Inference using model: {}'.format(checkpoint_path))
            saver.restore(sess, checkpoint_path)

            # pbar = tqdm(1000)
            for row_id in arange(0, rows-1, kernel_size):
                for col_id in arange(0, cols-1, kernel_size):
                    row_id, col_id = int(row_id), int(col_id)
                    # print(row_id, col_id)
                    tile = data_padding(data, row_id, col_id)
                    map = sess.run(return_map, feed_dict={input_image: np.expand_dims(tile[:, :, :FLAGS.num_of_channels], axis=0)})
                    row_step_size, col_step_size = int(kernel_size), int(kernel_size)
                    if row_id + kernel_size > rows - 1:
                        row_step_size = int(rows - 1 - row_id)
                    if col_id + kernel_size > cols - 1:
                        col_step_size = int(cols - 1 - col_id)
                    start_margin = int(expand_ratio * input_size)
                    if FLAGS.loss == 'dice':
                        scars_map = map[0, start_margin:start_margin + row_step_size, start_margin:start_margin + col_step_size] > 0.8
                        predict_map[row_id:row_id + row_step_size, col_id:col_id + col_step_size] = np.squeeze(scars_map, axis=-1)
                    else:
                        scars_map = map[0, start_margin:start_margin + row_step_size, start_margin:start_margin + col_step_size, 1] > 0.9
                        predict_map[row_id:row_id + row_step_size, col_id:col_id + col_step_size] = scars_map


                    ground_truth_map[row_id:row_id+row_step_size, col_id:col_id+col_step_size] = \
                        tile[start_margin:start_margin + row_step_size, start_margin:start_margin + col_step_size, -1]
                    pbar.update(1)
            io.imsave(dir_concat(FLAGS.HOME, ['output', FLAGS.task_name, 'predictions_{}.jpg'.format(FLAGS.epoch)]), predict_map)
            io.imsave(dir_concat(FLAGS.HOME, ['output', FLAGS.task_name, 'ground_truth_{}.jpg'.format(FLAGS.epoch)]), ground_truth_map)

    correct_pred = np.sum((predict_map[:, :] == ground_truth_map[:, :]) * ground_truth_map)
    pred = np.sum(predict_map) + np.sum(ground_truth_map) - correct_pred
    print('INFO: Overall accuracy on validation dataset is {:04f}'.format(correct_pred / pred))

if __name__ =='__main__':
    tf.app.run()


