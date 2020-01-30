import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2020-01-04_22_04_54"

from ISR.models import RDN
from ISR.train import Trainer
from ISR.utils.metrics import SSIM, PSNR
import numpy as np

seed = 1000
np.random.seed(seed)

lr_train_patch_size = 32
initial_scale = 2
scale = 2
hr_train_patch_size = lr_train_patch_size * scale

rdn = RDN(arch_params={'C': 3, 'D': 10, 'G': 64, 'G0': 64, 'x': scale})
loss_weights = {'generator': 0.1}
losses = {'generator': 'mae'}  # default = 'mse'
metric = {'generator': [SSIM, PSNR]}

log_dirs = {'logs': './logs',
            'weights': './weights'}

learning_rate = {'initial_value': 0.0001,
                 'decay_factor': 0.5,
                 'decay_frequency': 30}

flatness = {'min': 0.0,
            'max': 0.15,
            'increase': 0.01,
            'increase_frequency': 5}

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
    generator=rdn,
    train_dir='./Samples/300dpi/',
    valid_dir='./Samples/300dpi/',
    losses=losses,
    metrics=metric,
    loss_weights=loss_weights,
    learning_rate=learning_rate,
    flatness=flatness,
    log_dirs=log_dirs,
    weights_generator='weights/pre-train/rdn-C3-D10-G64-G064-x2/PSNR-driven/rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5',
    n_validation=16,
    seed=seed
)

trainer.train(
    epochs=100,
    epoch_size=1,
    batch_size=16,
    initial_scale=initial_scale,
    crop_height=lr_train_patch_size * scale,
    crop_weidth=lr_train_patch_size * scale,
    augment_train=aug_params_train,
    augment_valid=aug_params_valid
)
