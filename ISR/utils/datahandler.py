import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import yaml
import cv2

# Keras
import keras

# Albumentations
from albumentations import (
    Blur, Resize, GaussNoise, RandomFog, RandomGamma, RandomBrightnessContrast, Compose
)


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
                 method_rescale=cv2.INTER_AREA,
                 shuffle=True,
                 augment=None):
        self.dir = dir
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.init_scale = initial_scale
        self.scale = scale
        self.bord = bord
        self.crop_height = crop_height
        self.crop_weidth = crop_weidth
        self.method_rescale = method_rescale
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
        if self.epoch_size:
            return self.epoch_size
        else:
            return int(np.floor(len(self.images) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Generate one batch of Samples"""
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        images_hr = [self.resize(self.crop(cv2.imread(self.images[i])), self.init_scale) for i in
                     indexes]
        images_lr = [self.resize(image, self.scale) for image in images_hr]

        if self.augment:
            images_lr = [self.augmentor(image) for image in images_lr]

        return [i / 255 for i in images_hr], [i / 255 for i in images_lr]

    def crop(self, image):
        y = np.random.randint(self.bord, image.shape[0] - self.crop_height - self.bord)
        x = np.random.randint(self.bord, image.shape[1] - self.crop_weidth - self.bord)
        return image[y:y + self.crop_height, x:x + self.crop_weidth]

    def resize(self, image, scale):
        aug = Resize(image.shape[0] // scale, image.shape[1] // scale, self.method_rescale)
        return aug(image=image)['image']

    def get_dict_parameters(self):
        return self.settings

    def save_setting(self, path):
        yaml.dump(self.settings, open(path, 'w'))

    def augmentor(self, image):
        """Apply Samples augmentation"""
        aug = Compose([
            RandomBrightnessContrast(brightness_limit=self.augment["RandomBrightnessContrast"]["brightness_limit"],
                                     contrast_limit=self.augment["RandomBrightnessContrast"]["contrast_limit"],
                                     brightness_by_max=self.augment["RandomBrightnessContrast"]["brightness_by_max"],
                                     always_apply=self.augment["RandomBrightnessContrast"]["always_apply"],
                                     p=self.augment["RandomBrightnessContrast"]["p"]),
            Blur(blur_limit=self.augment["Blur"]["blur_limit"],
                 p=self.augment["Blur"]["p"]),
            RandomGamma(gamma_limit=self.augment["RandomGamma"]["gamma_limit"],
                        eps=self.augment["RandomGamma"]["eps"],
                        always_apply=self.augment["RandomGamma"]["always_apply"],
                        p=self.augment["RandomGamma"]["p"]),
            RandomFog(fog_coef_lower=self.augment["RandomFog"]["fog_coef_lower"],
                      fog_coef_upper=self.augment["RandomFog"]["fog_coef_upper"],
                      alpha_coef=self.augment["RandomFog"]["alpha_coef"],
                      always_apply=self.augment["RandomFog"]["always_apply"],
                      p=self.augment["RandomFog"]["p"]),
            GaussNoise(var_limit=self.augment["GaussNoise"]["var_limit"],
                       mean=self.augment["GaussNoise"]["mean"],
                       always_apply=self.augment["GaussNoise"]["always_apply"],
                       p=self.augment["GaussNoise"]["p"])
        ])
        return aug(image=image)['image']


if __name__ == "__main__":
    BATCH_SIZE = 16
    img_paths = './Samples/300dpi'
    path_crop = ['example_batch_hr.jpeg', 'example_batch_lr.jpeg']

    d = DataHandler(img_paths, epoch_size=2000, initial_scale=2, scale=2, batch_size=BATCH_SIZE, shuffle=True)
    images = d.__getitem__(1)
    for index in range(2):
        fig = plt.figure(figsize=(40, 40))
        columns = 4
        rows = 4

        for i in range(1, len(images[index]) + 1):
            ax = fig.add_subplot(rows, columns, i)
            plt.imshow(images[index][i - 1])

        plt.savefig(path_crop[index], dpi=300)
