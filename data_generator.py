from data_loader import load_images_list_with_truth
import numpy as np
from keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, image_filenames, batch_size):
        self._image_filenames = image_filenames
        self._batch_size = batch_size
        self._len = int(np.ceil(len(self._image_filenames) / float(self._batch_size)))

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        batch_x = self._image_filenames[idx * self._batch_size:(idx + 1) * self._batch_size]

        return load_images_list_with_truth(images_list=batch_x, normalize=True)
