from __future__ import division
import numpy as np
from os.path import join
from utils import load_tiff, write_txt

proj_home = '/media/data1/LanDCNN'
dataset_home = join(proj_home, 'dataset')
input_home = join(dataset_home, 'original_split')
output_home = join(dataset_home, 'preprocessed')


if __name__ == '__main__':
    for task in ['training', 'validation']:
        print('INFO: Processing for {} dataset.'.format(task))
        print('INFO: Loading datasets...')
        task_input_home = join(input_home, task)
        aerial_img_pre = load_tiff(join(task_input_home, 'aerial_image_pre_{}.tif'.format(task)))
        aerial_img_post = load_tiff(join(task_input_home, 'aerial_image_post_{}.tif'.format(task)))
        dtm = load_tiff(join(task_input_home, 'DTM_{}.tif'.format(task)))
        landslide_labels = load_tiff(join(task_input_home, 'landslide_labels_{}.tif'.format(task)))
        landslide_masks = load_tiff(join(task_input_home, 'landslide_masks_{}.tif'.format(task)))
        building_labels = load_tiff(join(task_input_home, 'building_labels_{}.tif'.format(task)))
        building_masks = load_tiff(join(task_input_home, 'building_masks_{}.tif'.format(task)))
        print("INFO: Loading completed.")

        print(aerial_img_pre.shape)
        print(aerial_img_post.shape)
        print(dtm.shape)
        print(landslide_labels.shape)
        print(landslide_masks.shape)
        print(building_labels.shape)
        print(building_masks.shape)

        rows = min(aerial_img_pre.shape[0],
                   aerial_img_post.shape[0],
                   dtm.shape[0],
                   landslide_labels.shape[0],
                   landslide_masks.shape[0],
                   building_labels.shape[0],
                   building_masks.shape[0])

        cols = min(aerial_img_pre.shape[1],
                   aerial_img_post.shape[1],
                   dtm.shape[1],
                   landslide_labels.shape[1],
                   landslide_masks.shape[1],
                   building_labels.shape[1],
                   building_masks.shape[1])

        landslide_labels[landslide_labels[:rows, :cols, 0] != 0] = 1
        landslide_masks[landslide_masks[:rows, :cols, 0] != 0] = 1
        building_labels[building_labels[:rows, :cols, 0] != 0] = 1
        building_masks[building_masks[:rows, :cols, 0] != 0] = 1

        labels = np.zeros([rows, cols, 1])
        # labels[landslide_labels[:rows, :cols, 0] == 1, 0] = 1
        labels[landslide_masks[:rows, :cols, 0] == 1, 0] = 0
        labels[building_labels[:rows, :cols, 0] == 1, 0] = 1
        labels[building_masks[:rows, :cols, 0] == 1, 0] = -1

        dataset = np.concatenate([aerial_img_pre[:rows, :cols, :],
                                  aerial_img_post[:rows, :cols, :],
                                  dtm[:rows, :cols, :],
                                  labels[:rows, :cols, :]], axis=-1)

        dataset_mean = np.mean(dataset[..., :-1], axis=(0, 1))
        dataset_std = np.std(dataset[..., :-1], axis=(0, 1))

        print('INFO: Saving {} dataset.'.format(task))
        np.save(join(output_home, '{}.npy'.format(task)), dataset)
        np.save(join(output_home, '{}_mean.npy'.format(task)), dataset_mean)
        np.save(join(output_home, '{}_std.npy'.format(task)), dataset_std)

    print("INFO: Preprocess completed.")
