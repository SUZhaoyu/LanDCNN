from __future__ import division
import os
import numpy as np
from os.path import join
import tensorflow as tf

tf.app.flags.DEFINE_string('HOME', '/media/data1/ENTLI', '')
tf.app.flags.DEFINE_string('task_name', 'test', '')
tf.app.flags.DEFINE_boolean('DTM', True, '')
tf.app.flags.DEFINE_boolean('global_standardize', True, '')

FLAGS = tf.app.flags.FLAGS
dataset_home = join(FLAGS.HOME, 'dataset')
input_home = join(dataset_home, 'input')
output_home = join(dataset_home, FLAGS.task_name)

from utils import load_tiff, write_txt



if __name__ == '__main__':

    print('INFO: Processing for training dataset.')
    train_home = join(input_home, 'train')
    DTM_path = join(train_home, 'DTM.tif')
    prior_path = join(train_home, 'prior_RGB.tif')
    posterior_path = join(train_home, 'posterior_RGB.tif')
    label_path = join(train_home, 'scars.tif')

    print('INFO: Loading datasets.')
    DTM = load_tiff(DTM_path)
    prior_RGB = load_tiff(prior_path)
    posterior_RGB = load_tiff(posterior_path)
    label = load_tiff(label_path)
    # TODO: Now only consider the landslide
    label[label[:, :, 0] != 0] = 1

    if FLAGS.DTM:
        rows = min(DTM.shape[0], prior_RGB.shape[0], posterior_RGB.shape[0], label.shape[0])
        cols = min(DTM.shape[1], prior_RGB.shape[1], posterior_RGB.shape[1], label.shape[1])
    else:
        rows = min(prior_RGB.shape[0], posterior_RGB.shape[0], label.shape[0])
        cols = min(prior_RGB.shape[1], posterior_RGB.shape[1], label.shape[1])
    if FLAGS.DTM:
        train_dataset = np.concatenate([prior_RGB[:rows, :cols, :], posterior_RGB[:rows, :cols, :], DTM[:rows, :cols, :], label[:rows, :cols, :]], axis=-1)
    else:
        train_dataset = np.concatenate([prior_RGB[:rows, :cols, :], posterior_RGB[:rows, :cols, :], label[:rows, :cols, :]], axis=-1)


    try:
        os.mkdir(output_home)
    except :
        print('WARNING: {} already exists, deleted.'.format(output_home))
        os.system('rm -r {}'.format(output_home))
        os.mkdir(output_home)

    if FLAGS.global_standardize:
        print('INFO: Standardizing training dataset.')
        for i in range(train_dataset.shape[-1] - 1):
            mean, std = np.mean(train_dataset[:, :, i]), np.std(train_dataset[:, :, i])
            write_txt(join(output_home, 'train.txt'), [mean, std])

    print('INFO: Saving training dataset.')
    np.save(join(output_home, 'train.npy'), train_dataset)

    print('INFO: Processing for validation dataset.')
    valid_home = join(input_home, 'valid')
    DTM_path = join(valid_home, 'DTM.tif')
    prior_path = join(valid_home, 'prior_RGB.tif')
    posterior_path = join(valid_home, 'posterior_RGB.tif')
    label_path = join(valid_home, 'scars.tif')
    void_path = join(valid_home, 'void.tif')

    print('INFO: Loading datasets.')
    DTM = load_tiff(DTM_path)
    prior_RGB = load_tiff(prior_path)
    posterior_RGB = load_tiff(posterior_path)
    label = load_tiff(label_path)

    if FLAGS.DTM:
        rows = min(DTM.shape[0], prior_RGB.shape[0], posterior_RGB.shape[0], label.shape[0])
        cols = min(DTM.shape[1], prior_RGB.shape[1], posterior_RGB.shape[1], label.shape[1])
    else:
        rows = min(prior_RGB.shape[0], posterior_RGB.shape[0], label.shape[0])
        cols = min(prior_RGB.shape[1], posterior_RGB.shape[1], label.shape[1])
    if FLAGS.DTM:
        valid_dataset = np.concatenate([prior_RGB[:rows, :cols, :], posterior_RGB[:rows, :cols, :], DTM[:rows, :cols, :], label[:rows, :cols, :]], axis=-1)
    else:
        valid_dataset = np.concatenate([prior_RGB[:rows, :cols, :], posterior_RGB[:rows, :cols, :], label[:rows, :cols, :]], axis=-1)

    if FLAGS.global_standardize:
        print('INFO: Standardizing validation dataset.')
        for i in range(valid_dataset.shape[-1] - 1):
            mean, std = np.mean(valid_dataset[:, :, i]), np.std(valid_dataset[:, :, i])
            write_txt(join(output_home, 'valid.txt'), [mean, std])


    print('INFO: Saving validation dataset.')
    np.save(join(output_home, 'valid.npy'), valid_dataset)

    print('INFO: Preprocess completed.')
    print('\n')
    print('----------------Dataset INFO------------------')
    print('     Training Dataset Size: {}'.format(train_dataset.shape[:2]))
    print('     Validation Dataset Size: {}'.format(valid_dataset.shape[:2]))
    print('     Use DTM: {}'.format(FLAGS.DTM))
    print('     Input Channels: {}'.format(train_dataset.shape[-1]-1))







