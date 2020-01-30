from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, Input, Lambda, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam


class Discriminator:
    def __init__(self, patch_size, kernel_size=3):
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.model = self.build_disciminator()
        optimizer = Adam(0.0002, 0.5)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])
        self.model.name = 'discriminator'
        self.name = 'srgan-large'

    def normalize_m11(self, x):
        """Normalizes RGB images to [-2020-01-04_22_04_54, 2020-01-04_22_04_54]."""
        return x / 127.5 - 1

    def discriminator_block(self, x_in, num_filters, strides=1, batchnorm=True, momentum=0.8):
        x = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(x_in)
        if batchnorm:
            x = BatchNormalization(momentum=momentum)(x)
        return LeakyReLU(alpha=0.2)(x)

    def build_disciminator(self, num_filters=64):
        x_in = Input(shape=(self.patch_size, self.patch_size, 3))
        x = Lambda(self.normalize_m11)(x_in)

        x = self.discriminator_block(x, num_filters, batchnorm=False)
        x = self.discriminator_block(x, num_filters, strides=2)

        x = self.discriminator_block(x, num_filters * 2)
        x = self.discriminator_block(x, num_filters * 2, strides=2)

        x = self.discriminator_block(x, num_filters * 4)
        x = self.discriminator_block(x, num_filters * 4, strides=2)

        x = self.discriminator_block(x, num_filters * 8)
        x = self.discriminator_block(x, num_filters * 8, strides=2)

        x = Flatten()(x)

        x = Dense(1024)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1, activation='sigmoid')(x)

        return Model(x_in, x)

# ToDo: Add discriminator BigGAN
# https://github.com/manicman1999/Keras-BiGAN/blob/bb7583fe61e4c56604ca95926c6e8ddf1057d16d/bigan.py#L159
