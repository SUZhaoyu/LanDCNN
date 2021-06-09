from __future__ import division
import numpy as np
import os
from os.path import join
from random import choice
from skimage import io
import time
import cv2
import copy
from tqdm import tqdm
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


num_class = 2
zoom_scale = [0.9, 1.1]
color_scale = [0.90, 1.10]
input_data = np.load(join('/home/tan/tony/entli/dataset/preprocessed/training.npy'))
input_size = 512

def get_data_tiles(data):
    rows, cols = data.shape[:2]
    sample_scale_h = np.clip(np.random.randn() / 12. + 1., zoom_scale[0], zoom_scale[1])
    sample_scale_w = np.clip(np.random.randn() / 12. + 1., zoom_scale[0], zoom_scale[1])
    # sample_scale = 1.
    size_h = int(input_size * sample_scale_h)
    size_w = int(input_size * sample_scale_w)
    selected_rows = np.random.randint(rows - size_h)
    selected_cols = np.random.randint(cols - size_w)
    return copy.deepcopy(data[selected_rows:selected_rows+size_h, selected_cols:selected_cols+size_w, :])

def zoom(tile):
    output_tile = np.zeros([input_size, input_size, tile.shape[-1]])
    output_tile[:, :, :-1] = cv2.resize(tile[:, :, :-1], (input_size, input_size), interpolation=cv2.INTER_AREA)
    output_tile[:, :, -1] = cv2.resize(tile[:, :, -1], (input_size, input_size), interpolation=cv2.INTER_NEAREST)
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
    aerial_img = tile[:, :, :3].astype(np.uint8)
    tile[:, :, :3] = hsv(lab(aerial_img)).astype(np.float32)

    return tile

def weight_mask(label_maps):
    w = np.ones((label_maps.shape[0], label_maps.shape[1]), dtype=np.float32)
    for i in np.arange(1, num_class):
        w[label_maps[:, :, i] != 0] =50.
    return w

def norm(tile):
    for i in range(tile.shape[2]):
        tile[:, :, i] -= np.mean(tile[:, :, i])
        if np.std(tile[:, :, i]) != 0:
            tile[:, :, i] /= np.std(tile[:, :, i])
        else:
            tile[:, :, i] = np.zeros_like(tile[:, :, i])
    return tile

def generator(batch_size):
    output_data, output_labels = [], []
    while True:
        tile = get_data_tiles(input_data)
        tile = zoom(tile)
        tile = flip(tile)
        tile = color_adjust(tile)
        tile = norm(tile)

        output_data.append(tile[:, :, :-1])
        output_labels.append(tile[:, :, -1])

        if len(output_data) == batch_size:
            yield output_data, output_labels
            output_data, output_labels = [], []


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

if __name__ =='__main__':
    output_home = '/home/tan/tony/entli/dataset/preprocessed/viz'
    for i in tqdm(range(50)):
        imgs, labels = next(generator(1))
        aerial_img = imgs[0][:, :, :3]
        DTM_img = imgs[0][:, :, -1]
        DTM_img -= np.min(DTM_img)
        DTM_img /= np.max(DTM_img) + 1e-7
        io.imsave(join(output_home, 'aerial_img_{}.jpg'.format(i)), aerial_img.astype(np.uint8))
        io.imsave(join(output_home, 'DTM_{}.jpg'.format(i)), DTM_img)
        io.imsave(join(output_home, 'label_{}.jpg').format(i), np.squeeze(labels[0] / 2.))





