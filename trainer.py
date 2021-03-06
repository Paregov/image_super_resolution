import os
import time
import logging
from enum import Enum
from abc import abstractmethod, ABC
from utils import check_path_exists
from keras.layers import Input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_loader import load_images_list_with_truth


logger = logging.getLogger('image_super_resolution')


# loss_function
# optimizer
# weights - [ G, D]
# load_weights
# train_times - [G, D]

# We need to pass the training, validation and test data
# This will be done through DataProvider
# data_providers = {DataType.Training: training,
#                   DataType.Validation: validation,
#                   DataType.Test: test}

# Models that we want to use. The idea to pass the models as well is so we cantrain with different initializerz
# models = {'generator': None, 'discriminator': None}

# Paths to the weights
# weights = {'generator': '', 'discriminator': ''}

# Training time in hours
# train_times = {'generator': '', 'discriminator': ''}

# arguments = {'kernel_initializer': 'glorot_uniform', 'loss_function': '',
#              'optimizer': 'adam', 'weights': weights, 'load_weights': True,
#              'train_times': train_times}


class DataType(Enum):
    Training = 1
    Validation = 2
    Test = 3


class ModelType(Enum):
    Generator = 1
    Discriminator = 2
    GAN = 3


class DataProvider(ABC):
    """
    Base class to provide data for the trainers.
    """
    @abstractmethod
    def len(self):
        return 0

    @abstractmethod
    def batches_len(self):
        return 0

    @abstractmethod
    def get_batch(self, idx):
        pass

    @abstractmethod
    def get_batch_by_indexes(self, indexes):
        pass


class FilesDataProvider(DataProvider):
    def __init__(self, image_file_names, batch_size):
        self._image_file_names = image_file_names
        self._batch_size = batch_size
        self._len = len(self._image_file_names)
        self._batches_len = int(np.ceil(len(self._image_file_names) / float(self._batch_size)))

    def len(self):
        return self._len

    def batches_len(self):
        return self._batches_len

    def get_batch(self, idx):
        batch_x = self._image_file_names[idx * self._batch_size:(idx + 1) * self._batch_size]

        return load_images_list_with_truth(images_list=batch_x, normalize=True)

    def get_batch_by_indexes(self, indexes):
        batch_x = np.array(self._image_file_names)[tuple(indexes)]

        return load_images_list_with_truth(images_list=batch_x, normalize=True)


class InMemoryDataProvider(DataProvider):
    def __init__(self, X, y, batch_size):
        self._X = X
        self._y = y
        self._batch_size = batch_size
        self._len = len(self._X)
        self._batches_len = int(np.ceil(len(self._X) / float(self._batch_size)))

    def len(self):
        return self._len

    def batches_len(self):
        return self._batches_len

    def get_batch(self, idx):
        return self._X[idx * self._batch_size:(idx + 1) * self._batch_size],\
               self._y[idx * self._batch_size:(idx + 1) * self._batch_size]

    def get_batch_by_indexes(self, indexes):
        return np.array(self._X)[tuple(indexes)],\
               np.array(self._y)[tuple(indexes)]


class GANTrainer:
    """
    GAN training class. It plots the result of the training on a specified period of training epochs.
    """
    def __init__(self, models, optimizers, losses, loss_weights, weights, load_weights, data_providers,
                 compare_path='.', **kwargs):
        """
        Initialization function for the GANTrainer.

        :param models:
        :param optimizers:
        :param losses:
        :param loss_weights:
        :param weights:
        :param load_weights:
        :param data_providers:
        """
        self._generator = models['generator']
        self._discriminator = models['discriminator']
        self._generator_weights = weights['generator']
        self._discriminator_weights = weights['discriminator']
        self._load_weights = load_weights

        self._training = data_providers[DataType.Training]
        # self._validation = data_providers[DataType.Validation]
        self._test = data_providers[DataType.Test]

        self._optimizer_g = optimizers['generator']
        self._optimizer_d = optimizers['discriminator']
        self._optimizer_gan = optimizers['gan']

        self._loss_g = losses['generator']
        self._loss_d = losses['discriminator']
        self._loss_gan = losses['gan']

        self._compare_path = compare_path

        self._compile_generator()
        self._compile_discriminator()

        X, y = self._training.get_batch(1)
        self._image_shape = X.shape[1:]

        self._gan = self.create_gan()

    def _compile_generator(self):
        self._generator.compile(optimizer=self._optimizer_g, loss=self._loss_g, metrics=['accuracy'])

    def _compile_discriminator(self):
        self._discriminator.compile(optimizer=self._optimizer_d, loss=self._loss_d, metrics=['accuracy'])

    def plot_generated_images(self, epoch, examples=10, dim=(2, 5), figsize=(15, 10), base_path='.'):
        images = self._test.get_batch_by_indexes(np.random.randint(low=0, high=self._test.len(), size=examples))
        generated_images = self._generator.predict(images)
        plt.figure(figsize=figsize)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i + 1)
            plt.imshow((generated_images[i] * 255).astype(np.uint8))
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('gan_generated_image %d.png' % epoch)

    def plot_images_for_compare(self, epoch, examples=7, base_path='.', base_name='compare_images_for_epoch_'):
        dim = (examples, 3)
        figsize = (examples, examples*2)

        indexes = [np.random.randint(low=0, high=self._test.len(), size=examples)]
        images, real_images = self._test.get_batch_by_indexes(indexes)
        generated_images = self._generator.predict(images)

        plt.figure(figsize=figsize)
        sub_plot = 0
        for i in range(generated_images.shape[0]):
            for a in range(3):
                if a == 0:
                    # Low resolution
                    plt.subplot(dim[0], dim[1], sub_plot + 1)
                    plt.imshow((images[i] * 255).astype(np.uint8), interpolation='nearest', aspect='equal')
                if a == 1:
                    # Generated image
                    plt.subplot(dim[0], dim[1], sub_plot + 1)
                    plt.imshow((generated_images[i] * 255).astype(np.uint8), interpolation='nearest', aspect='equal')
                if a == 2:
                    # High resolution
                    plt.subplot(dim[0], dim[1], sub_plot + 1)
                    plt.imshow((real_images[i] * 255).astype(np.uint8), interpolation='nearest', aspect='equal')
                plt.axis('off')
                sub_plot += 1

        plt.tight_layout()
        image_name = '{}{:06d}.png'.format(base_name, epoch)
        image_path = os.path.join(base_path, image_name)
        logger.debug('Saving for compare: {}'.format(image_path))
        plt.savefig(image_path)

    def create_gan(self):
        self._discriminator.trainable = False

        inputs = Input(self._image_shape)
        generated_images = self._generator(inputs)
        outputs = self._discriminator(generated_images)

        gan = Model(inputs=inputs, outputs=outputs)
        gan.compile(loss=self._loss_gan, optimizer=self._optimizer_gan)

        return gan

    def save_models_weights(self):
        self._generator.save_weights(self._generator_weights)
        self._discriminator.save_weights(self._discriminator_weights)

    def load_models_weights(self):
        if check_path_exists(self._generator_weights):
            print('Loading generator weights')
            self._generator.load_weights(self._generator_weights)
        else:
            print('No weights file for the generator to be loaded.')

        if check_path_exists(self._discriminator_weights):
            print('Loading discriminator weights')
            self._discriminator.load_weights(self._discriminator_weights)
        else:
            print('No weights file for the discriminator to be loaded.')

    def train(self, epochs=1, batch_size=32, epochs_between_plots=20, epochs_between_saves=100, max_train_time=1):
        """
        Train the GAN.

        Call this function to start training of the GAN.

        :param epochs: Number of epochs to train the model.
        :param batch_size: How many images to use per epoch. Default is 32.
        :param epochs_between_plots: Specify the number of epochs between each plotting of the images.
        :param epochs_between_saves: Specify the number of epochs between saving the weights of the models.
        :param max_train_time: Maximum time to train the model in minutes
        :return: None
        """
        if self._load_weights:
            self.load_models_weights()

        start_time = time.time()
        for e in range(1, epochs + 1):
            logger.info("Epoch %d" % e)
            for _ in tqdm(range(batch_size)):
                # Get images as input for the generator.
                indexes = [np.random.randint(low=0, high=self._training.len(), size=batch_size)]
                X_gen, __ = self._training.get_batch_by_indexes(indexes)
                generated_images = self._generator.predict(X_gen)

                # Get a random set of  real images
                indexes = [np.random.randint(low=0, high=self._training.len(), size=batch_size)]
                __, real_images = self._training.get_batch_by_indexes(indexes)

                # Construct different batches of real and fake data
                X_dis = np.concatenate([real_images, generated_images])

                # Labels for generated and real data
                y_dis = np.zeros(2 * batch_size)
                y_dis[:batch_size] = 0.9

                # Pre train discriminator on  fake and real data  before starting the gan.
                self._discriminator.trainable = True
                d_loss = self._discriminator.train_on_batch(X_dis, y_dis)

                # Tricking the input of the Generator as real data
                indexes = [np.random.randint(low=0, high=self._training.len(), size=batch_size)]
                X_gan, __ = self._training.get_batch_by_indexes(indexes)
                y_gan = np.ones(batch_size)

                # During the training of gan,
                # the weights of discriminator should be fixed.
                # We can enforce that by setting the trainable flag.
                self._discriminator.trainable = False

                # Training  the GAN by alternating the training of the Discriminator
                # and training the chained GAN model with Discriminator’s weights friezed.
                a_loss = self._gan.train_on_batch(X_gan, y_gan)

            # Plot the progress
            logger.info("%d [D loss: %f, acc.: %.2f%%] [A loss: %f]" % (e, d_loss[0], 100 * d_loss[1], a_loss))

            if e == 1 or e % epochs_between_plots == 0:
                self.plot_images_for_compare(epoch=e, base_path=self._compare_path)

            if epochs_between_saves > 0 and e % epochs_between_saves == 0:
                self.save_models_weights()

            current_time = time.time()
            if ((current_time - start_time)/60) > max_train_time:
                logger.info('Model {} has been trained for the max_train_time ({})'.format('NAME', max_train_time))
                break


class GANGridTrainer:
    """

    """
    def __init__(self, param_grid, weights_path, load_weights, data_providers, compare_path, **kwargs):
        """
        Perform a grid search on the GAN training.

        :param param_grid:
        :param weights_path:
        :param load_weights:
        :param tensors:
        :param compare_path:
        :param kwargs:
        """
        self._param_grid = param_grid
        self._weight_path = weights_path
        self._load_weights = load_weights
        self._data_providers = data_providers
        self._compare_path = compare_path

    # How to save the weights for the grid?
    # May be keep a map how to

    # Go through all of the combinations
    def train(self):
        for m in self._param_grid['models']:
            for o in self._param_grid['optimizers']:
                for l in self._param_grid['losses']:
                    # TODO: Save the combination that is currently used
                    # TODO: Generate full paths for the weights here
                    t = GANTrainer(models=m, optimizers=o, losses=l, loss_weights=None,
                                   weights=None, load_weights=self._load_weights, data_providers=self._data_providers)
                    t.train(1000)

