import mkl
import numpy as np
import cv2
import copy
# import imgaug.augmenters as iaa

mkl.set_num_threads(1)

num_class = 3
zoom_scale = [0.9, 1.1]
color_scale = [0.90, 1.10]
input_size = 512

def get_data_tiles(data):
    rows, cols = data.shape[:2]
    sample_scale_h = np.clip(np.random.randn() / 12. + 1., zoom_scale[0], zoom_scale[1])
    sample_scale_w = np.clip(np.random.randn() / 12. + 1., zoom_scale[0], zoom_scale[1])
    # sample_scale_w = 1.
    # sample_scale_h = 1.
    size_h = int(input_size * sample_scale_h)
    size_w = int(input_size * sample_scale_w)
    selected_rows = np.random.randint(rows - size_h)
    selected_cols = np.random.randint(cols - size_w)
    return copy.deepcopy(data[selected_rows:selected_rows+size_h, selected_cols:selected_cols+size_w, :])

def zoom(tile):
    # https://zhuanlan.zhihu.com/p/45030867
    output_tile = np.zeros([input_size, input_size, tile.shape[-1]])
    output_tile[:, :, :-1] = cv2.resize(tile[:, :, :-1], (input_size, input_size), interpolation=cv2.INTER_LINEAR)
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
