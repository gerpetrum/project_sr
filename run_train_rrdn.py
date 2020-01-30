import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2020-01-04_22_04_54"

from ISR.models import RRDN
from ISR.models import Discriminator
from ISR.models import Cut_VGG19
from ISR.train import Trainer
from ISR.utils.metrics import PSNR, PSNR_Y, SSIM
import numpy as np

seed = 1000
np.random.seed(seed)

lr_train_patch_size = 24
layers_to_extract = [5, 9]
initial_scale = 1
scale = 4
hr_train_patch_size = lr_train_patch_size * scale

rrdn = RRDN(arch_params={'C': 4, 'D': 3, 'G': 32, 'G0': 32, 'T': 10, 'x': scale}, patch_size=lr_train_patch_size)
f_ext = Cut_VGG19(patch_size=hr_train_patch_size, layers_to_extract=layers_to_extract)
discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)

loss_weights = {
    'generator': 0.1,
    'feature_extractor': 0.0833,
    'discriminator': 0.01
}

losses = {
    'generator': 'mae',
    'feature_extractor': 'mse',
    'discriminator': 'binary_crossentropy'
}

metrics = {'generator': [PSNR, PSNR_Y, SSIM]}
log_dirs = {'logs': './logs', 'weights': './weights'}
learning_rate = {'initial_value': 0.0001, 'decay_factor': 0.5, 'decay_frequency': 30}
flatness = {'min': 0.0, 'max': 0.15, 'increase': 0.01, 'increase_frequency': 5}

aug_params_train = {"RandomBrightnessContrast": {"brightness_limit": 0.2,
                                                 "contrast_limit": 0.2,
                                                 "brightness_by_max": True,
                                                 "always_apply": False,
                                                 "p": 0.25},
                    "Blur": {"blur_limit": 3,
                             "p": 0.25},
                    "RandomGamma": {"gamma_limit": (80, 120),
                                    "eps": 1e-07,
                                    "always_apply": False,
                                    "p": 0.1},
                    "RandomFog": {"fog_coef_lower": 0.3,
                                  "fog_coef_upper": 1,
                                  "alpha_coef": 0.08,
                                  "always_apply": False,
                                  "p": 0.1},
                    "GaussNoise": {"var_limit": (10.0, 50.0),
                                   "mean": 0,
                                   "always_apply": False,
                                   "p": 0.1}
                    }
aug_params_valid = aug_params_train

trainer = Trainer(
    generator=rrdn,
    discriminator=discr,
    feature_extractor=f_ext,
    train_dir='./Samples/300dpi/',
    valid_dir='./Samples/300dpi/',
    loss_weights=loss_weights,
    learning_rate=learning_rate,
    metrics=metrics,
    flatness=flatness,
    dataname='name_dataset',
    log_dirs=log_dirs,
    weights_generator='weights/pre-train/rrdn-C4-D3-G32-G032-T10-x4/Perceptual/rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5',
    weights_discriminator='weights/pre-train/rrdn-C4-D3-G32-G032-T10-x4/Perceptual/DiscriminatorSRGAN.h5',
    n_validation=2,
    seed=seed
)

trainer.train(
    epochs=1,
    epoch_size=1,
    batch_size=2,
    initial_scale=initial_scale,
    crop_height=lr_train_patch_size * scale,
    crop_weidth=lr_train_patch_size * scale,
    augment_train=aug_params_train,
    augment_valid=aug_params_valid
)
