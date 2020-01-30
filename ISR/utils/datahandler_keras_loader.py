import numpy as np
from glob import glob
import yaml
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

# Keras
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


class DataHandler(keras.utils.Sequence):
    """Generates Samples for Keras"""

    def __init__(self,
                 dir,
                 epoch_size,
                 batch_size=64,
                 initial_scale=1,
                 scale=4,
                 bord=200,
                 crop_height=200,
                 crop_weidth=200,
                 shuffle=True,
                 augment=False):
        self.dir = dir
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.init_scale = initial_scale
        self.scale = scale
        self.bord = bord
        self.crop_height = crop_height
        self.crop_weidth = crop_weidth
        self.shuffle = shuffle
        self.augment = augment
        self.settings = locals()
        _ = self.settings.pop('self')

        self.images = glob(dir + '/*.jpg')
        if self.epoch_size > len(self.images) // self.batch_size:
            self.epoch_size = None
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.images) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    @staticmethod
    def get_image_array(path):
        return img_to_array(load_img(path))

    def __getitem__(self, index):
        """Generate one batch of Samples"""
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        images_hr = [self.resize(self.crop(self.get_image_array(self.images[index])), self.init_scale) / 255.0
                     for index in indexes]
        images_lr = [self.resize(image, self.scale) for image in images_hr]

        return images_hr, images_lr

    def crop(self, image):
        while True:
            y = np.random.randint(image.shape[0] - self.crop_height)
            x = np.random.randint(image.shape[1] - self.crop_weidth)
            crop = image[y:y + self.crop_height, x:x + self.crop_weidth]
            if crop[0].max() != crop[0].min() and \
                    crop[1].max() != crop[1].min() and \
                    crop[2].max() != crop[2].min():
                break
        return crop

    def resize(self, image, scale):
        return np.array([zoom(i, 1.0 / scale) for i in image.transpose([2, 0, 1])]).transpose((1, 2, 0))

    def get_dict_parameters(self):
        return self.settings

    def save_setting(self, path):
        yaml.dump(self.settings, open(path, 'w'))


if __name__ == "__main__":
    BATCH_SIZE = 16
    img_paths = '../Samples'
    path_crop = ['example_batch_hr.jpeg', 'example_batch_lr.jpeg']

    d = DataHandler(img_paths, scale=2, batch_size=BATCH_SIZE, shuffle=True)
    images = d.__getitem__(1)
    for index in range(2):
        fig = plt.figure(figsize=(40, 40))
        columns = 4
        rows = 4

        for i in range(1, len(images[index]) + 1):
            ax = fig.add_subplot(rows, columns, i)
            plt.imshow(images[index][i - 1])

        plt.savefig(path_crop[index], dpi=300)
