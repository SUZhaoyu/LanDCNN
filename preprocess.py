from __future__ import division
import os
import numpy as np
from os.path import join
from utils import load_tiff, write_txt
import tensorflow as tf

proj_home = '/home/tan/tony/entli'
dataset_home = join(proj_home, 'dataset')
input_home = join(dataset_home, 'input')
output_home = join(dataset_home, 'preprocessed')


if __name__ == '__main__':
    for task in ['validation', 'training']:
        print('INFO: Processing for {} dataset.'.format(task))
        print('INFO: Loading datasets...')
        task_input_home = join(input_home, task)
        aerial_img = load_tiff(join(task_input_home, 'aerial_image_{}.tif'.format(task)))
        dtm = load_tiff(join(task_input_home, 'DTM_{}.tif'.format(task)))
        landslide_labels = load_tiff(join(task_input_home, 'landslide_labels_{}.tif'.format(task)))
        landslide_masks = load_tiff(join(task_input_home, 'landslide_masks_{}.tif'.format(task)))
        building_labels = load_tiff(join(task_input_home, 'building_labels_{}.tif'.format(task)))
        building_masks = load_tiff(join(task_input_home, 'building_masks_{}.tif'.format(task)))
        print("INFO: Loading completed.")

        rows = min(aerial_img.shape[0],
                   dtm.shape[0],
                   landslide_labels.shape[0],
                   landslide_masks.shape[0],
                   building_labels.shape[0],
                   building_masks.shape[0])

        cols = min(aerial_img.shape[1],
                   dtm.shape[1],
                   landslide_labels.shape[1],
                   landslide_masks.shape[1],
                   building_labels.shape[1],
                   building_masks.shape[1])

        # TODO: Now only consider the landslide
        landslide_labels[landslide_labels[:rows, :cols, 0] != 0] = 1
        landslide_labels[landslide_masks[:rows, :cols, 0] != 0] = 1
        building_labels[building_labels[:rows, :cols, 0] != 0] = 1
        building_masks[building_masks[:rows, :cols, 0] != 0] = 1

        labels = np.zeros([rows, cols, 1])
        labels[landslide_labels[:rows, :cols, 0] == 1] = 1
        labels[building_labels[:rows, :cols, 0] == 1] = 2
        labels[building_masks[:rows, :cols, 0] == 1] = -1

        dataset = np.concatenate([aerial_img[:rows, :cols, :],
                                  dtm[:rows, :cols, :],
                                  labels[:rows, :cols, :]], axis=-1)

        print('INFO: Saving {} dataset.'.format(task))
        np.save(join(output_home, '{}.npy'.format(task)), dataset)

    print("INFO: Preprocess completed.")
