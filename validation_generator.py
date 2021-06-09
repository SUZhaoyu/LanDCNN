from __future__ import division
import numpy as np
from numpy import arange
from os.path import join
from skimage import io
import cv2
import tensorflow as tf

from utils import dir_concat

FLAGS = tf.app.flags.FLAGS

input_size = FLAGS.input_size
expand_ratio = FLAGS.expand_ratio
kernel_size = int(input_size * (1 - 2 * expand_ratio))

def data_padding(data, row_id, col_id):
    rows, cols = data.shape[:2]
    start_row_id = int(row_id - expand_ratio * input_size)
    start_col_id = int(col_id - expand_ratio * input_size)
    stop_row_id = int(start_row_id + input_size)
    stop_col_id = int(start_col_id + input_size)
    selected_tile = data[max(start_row_id, 0):min(stop_row_id, rows-1), max(start_col_id, 0):min(stop_col_id, cols-1), :]
    padding_top = max(start_row_id, 0) - start_row_id
    padding_bottom = stop_row_id - min(stop_row_id, rows-1)
    padding_left = max(start_col_id, 0) - start_col_id
    padding_right = stop_col_id - min(stop_col_id, cols - 1)
    if padding_left + padding_right + padding_bottom + padding_top != 0:
        return cv2.copyMakeBorder(selected_tile, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_REFLECT)
    else:
        return selected_tile