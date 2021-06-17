from __future__ import division
import numpy as np
import multiprocessing
import mkl
import time
from os.path import join
from tqdm import tqdm
mkl.set_num_threads(1)

from augment_utils import get_data_tiles, zoom, flip, color_adjust, norm


class Dataset(object):
    def __init__(self,
                 task,
                 batch_size=16,
                 queue_size=10,
                 num_worker=1,
                 home='/home/tan/tony/entli/dataset/preprocessed'):
        self.input_data = np.load(join(home, '{}.npy'.format(task)))
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.num_worker = num_worker
        self.threads = []
        self.q = multiprocessing.Queue(maxsize=self.queue_size)
        self.start()

    def start(self):
        for i in range(self.num_worker):
            thread = multiprocessing.Process(target=self.aug_process)
            thread.daemon = True
            self.threads.append(thread)
            thread.start()

    def stop(self):
        for i, thread in enumerate(self.threads):
            thread.terminate()
            thread.join()
            self.q.close()

    def aug_process(self):
        while True:
            try:
                if self.q.qsize() < self.queue_size:
                    output_data, output_labels = [], []
                    for _ in range(self.batch_size):
                        tile = get_data_tiles(self.input_data)

                        tile = zoom(tile)
                        tile = flip(tile)
                        tile = color_adjust(tile)
                        tile = norm(tile)

                        output_data.append(tile[:, :, :-1])
                        output_labels.append(tile[:, :, -1])

                    self.q.put([np.array(output_data), np.array(output_labels)])
                else:
                    time.sleep(0.05)
            except:
                self.stop()

    def train_generator(self):
        while True:
            if self.q.qsize() > 0:
                yield self.q.get()
            else:
                time.sleep(0.05)





if __name__ =='__main__':
    output_home = '/home/tan/tony/entli/dataset/preprocessed/viz'
    TrainingDataset = Dataset(task='training', num_worker=1)
    training_data_generator = TrainingDataset.train_generator()
    for i in tqdm(range(5000)):
        imgs, labels = next(training_data_generator)
        print(imgs.shape, labels.shape)
        # aerial_img = imgs[0][:, :, :3]
        # DTM_img = imgs[0][:, :, -1]
        # DTM_img -= np.min(DTM_img)
        # DTM_img /= np.max(DTM_img) + 1e-7
        # io.imsave(join(output_home, 'aerial_img_{}.jpg'.format(i)), aerial_img.astype(np.uint8))
        # io.imsave(join(output_home, 'DTM_{}.jpg'.format(i)), DTM_img)
        # io.imsave(join(output_home, 'label_{}.jpg').format(i), np.squeeze(labels[0] / 2.))





