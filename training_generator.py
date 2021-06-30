from __future__ import division
import numpy as np
import os
from os.path import join
from random import choice
from skimage import io
import time
import cv2
import copy
import tensorflow as tf

# The multi-thread wrapper idea was copied from https://github.com/argman/EAST
from thread_wrapper import GeneratorEnqueuer
from utils import dir_concat, read_info_from_txt



# =============TEST VARIABLES===============

# use_DTM = True
#
# tf.app.flags.DEFINE_string('gpu_list', '0,1,2', '')
# tf.app.flags.DEFINE_string('HOME', '/media/data1/ENTLI','')
# tf.app.flags.DEFINE_string('task_name', 'test','')
# tf.app.flags.DEFINE_string('phase', 'train', '')
# tf.app.flags.DEFINE_boolean('DTM', use_DTM, '')
# if use_DTM:
#     tf.app.flags.DEFINE_integer('num_of_channels', 6, '')
# else:
#     tf.app.flags.DEFINE_integer('num_of_channels', 7, '')
# tf.app.flags.DEFINE_integer('input_size', 512, '')
# tf.app.flags.DEFINE_boolean('continue_train', False , '')
# tf.app.flags.DEFINE_float('learning_rate', 0.0002, '')
# tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
# tf.app.flags.DEFINE_integer('batch_size_per_gpu', 10, '')
# tf.app.flags.DEFINE_integer('num_of_labels', 2, '')



FLAGS = tf.app.flags.FLAGS


zoom_scale = [0.8, 1.2]
color_scale = [0.90, 1.10]
task = 'validation'
dataset_home = join(FLAGS.HOME, 'dataset')
data = np.load(dir_concat(FLAGS.HOME, ['dataset', FLAGS.dataset_name, '{}.npy'.format(FLAGS.phase)]))
mean = np.load(dir_concat(FLAGS.HOME, ['dataset', FLAGS.dataset_name, '{}_mean.npy'.format(FLAGS.phase)]))
std = np.load(dir_concat(FLAGS.HOME, ['dataset', FLAGS.dataset_name, '{}_std.npy'.format(FLAGS.phase)]))
# means, stds = read_info_from_txt(dir_concat(FLAGS.HOME, ['dataset', FLAGS.dataset_name, 'train.txt']))
assert len(mean) == FLAGS.num_of_channels







def get_data_tiles(data):
    rows, cols = data.shape[:2]
    sample_scale = np.clip(np.random.randn() / 12. + 1., zoom_scale[0], zoom_scale[1])
    # sample_scale = 1.
    input_size = int(FLAGS.input_size * sample_scale)
    selected_rows = np.random.randint(rows - input_size)
    selected_cols = np.random.randint(cols - input_size)
    return copy.deepcopy(data[selected_rows:selected_rows+input_size, selected_cols:selected_cols+input_size, :])

def zoom(tile):
    source_rows, source_cols = tile.shape[:2]
    if source_rows == FLAGS.input_size:
        return tile
    output_tile = np.zeros([FLAGS.input_size, FLAGS.input_size, tile.shape[-1]])
    output_tile[:, :, :-1] = cv2.resize(tile[:, :, :-1], (FLAGS.input_size, FLAGS.input_size), interpolation=cv2.INTER_LINEAR)
    output_tile[:, :, -1] = cv2.resize(tile[:, :, -1], (FLAGS.input_size, FLAGS.input_size), interpolation=cv2.INTER_NEAREST)

    return output_tile

def flip(tile):
    if np.random.uniform() > 0.5:
        tile = np.flipud(tile)
    if np.random.uniform() > 0.5:
        tile = np.fliplr(tile)
    nb_rotations = np.random.randint(0, 4)
    tile = np.rot90(tile, nb_rotations)
    return tile

def lab(img):
    imglab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype("float32")
    l_factor = np.clip(np.random.randn() / 30. + 1., color_scale[0], color_scale[1])
    a_factor = np.clip(np.random.randn() / 30. + 1., color_scale[0], color_scale[1])
    b_factor = np.clip(np.random.randn() / 30. + 1., color_scale[0], color_scale[1])
    # print(l_factor, a_factor, b_factor)
    (l, a, b) = cv2.split(imglab)
    l = l * l_factor
    a = a * a_factor
    b = b * b_factor

    imglab = cv2.merge([l, a, b])
    imglab = np.clip(imglab, 0, 255)
    imgrgb = cv2.cvtColor(imglab.astype("uint8"), cv2.COLOR_LAB2RGB)

    return imgrgb


def hsv(img):
    imghsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype("float32")
    s_factor = np.clip(np.random.randn() / 30. + 1., color_scale[0], color_scale[1])
    (h, s, v) = cv2.split(imghsv)

    s *= s_factor
    s = np.clip(s, 0, 255)
    imghsv = cv2.merge([h, s, v])
    imgrgb = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2RGB)

    return imgrgb

def color_adjust(tile):
    earlier_RGB = tile[:, :, :3].astype(np.uint8)
    posterior_RGB = tile[:, :, 3:6].astype(np.uint8)
    tile[:, :, :3] = hsv(lab(earlier_RGB)).astype(np.float32)
    tile[:, :, 3:6] = hsv(lab(posterior_RGB)).astype(np.float32)

    return tile

def one_hot_converter(label):
    label_maps = np.zeros(shape=(label.shape[0], label.shape[1], FLAGS.num_of_labels), dtype=np.float32)
    label_maps[:, :, 0] = np.less(label, 0.5).astype(np.float32)
    label_maps[:, :, 1] = np.greater(label, 0.5).astype(np.float32)
    # for i in range(FLAGS.num_of_labels):
    #     label_maps[label[:, :] == i, i] = 1.
    return label_maps

def weight_mask(label_maps):
    w = np.ones((label_maps.shape[0], label_maps.shape[1]), dtype=np.float32)
    w[label_maps < -0.5] = 0.
    return w



def generator(batch_size):
    imgs, label_maps, weight = [], [], []
    while True:
        tile = get_data_tiles(data)
        tile = zoom(tile)
        tile = flip(tile)
        tile = color_adjust(tile)
        tile[:, :, :-1] -= mean
        tile[:, :, :-1] /= std
        # for i in range(FLAGS.num_of_channels):
        #     tile[:, :, i] -= means[i]
        #     tile[:, :, i] /= stds[i]
        tile_label_maps = one_hot_converter(tile[:, :, -1])
        tile_weight = weight_mask(tile[:, :, -1])
        if FLAGS.loss == 'dice':
            label_maps.append(np.expand_dims(tile[:, :, -1], axis=-1))
        else:
            label_maps.append(tile_label_maps)
        imgs.append(tile[:, :, :-1])
        weight.append(tile_weight)

        if len(imgs) == batch_size:
            yield imgs, label_maps, weight
            imgs, label_maps, weight = [], [], []


def get_batch(num_workers, max_queue_size=10, **kwargs):
    try:

        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=max_queue_size, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()

# if __name__ =='__main__':
#     output_home = join(FLAGS.HOME, 'demo_output')
#     i = 0
#     while i < 50:
#         imgs, label_maps = generator(data, 1).__next__()
#         before_img = imgs[0][:, :, :3]
#         after_img = imgs[0][:, :, 3:6]
#         io.imsave(join(output_home, 'before_{}.jpg'.format(i)), before_img.astype(np.uint8))
#         io.imsave(join(output_home, 'after_{}.jpg'.format(i)), after_img.astype(np.uint8))
#         if FLAGS.DTM:
#             DTM_img = imgs[0][:, :, -1]
#             io.imsave(join(output_home, 'DTM_{}.jpg'.format(i)), DTM_img/np.max(DTM_img))
#         io.imsave(join(output_home, 'label_{}.jpg').format(i), np.squeeze(label_maps[0][:, :, 0]))
#         i += 1





