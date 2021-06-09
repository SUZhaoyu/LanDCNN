import os
from os.path import join
import rasterio
import numpy as np

def dir_concat(home, sub_list):
    target = home
    for dir in sub_list:
        target = join(target, dir)

    return target

def load_tiff(path):
    with rasterio.open(path, 'r') as r:
        return np.transpose(r.read(), axes=[1, 2, 0]).astype(np.float32)

def write_txt(path, values):
    str_values=''
    for value in values:
        str_values += (str(value)+',')
    with open(path, 'a+') as f:
        f.write(str_values[:-1] + '\n')

def read_info_from_txt(path):
    means, stds = [], []
    with open(path) as f:
        for line in f:
            if len(line) != 0:
                means.append(float(line.split(',')[0]))
                stds.append(float(line.split(',')[1]))
            else:
                break
    if len(means) == 0 or len(stds) == 0:
        raise ValueError('ERROR: There is no data extracted from {}'.format(path))
    return means, stds
