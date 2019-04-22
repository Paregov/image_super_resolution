import os
import logging
from utils import check_path_exists
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


logger = logging.getLogger('image_super_resolution')

class Trainer:
    def __init__(self, model, weights, load_weights, tensors, train_time=1, batch_size=32):
        self._model = model
        self._weights = weights
        self._load_weights = load_weights

        self._train_X = tensors['train']['X']
        self._train_y = tensors['train']['y']
        self._valid_X = tensors['valid']['X']
        self._valid_y = tensors['valid']['y']
        self._test_X = tensors['test']['X']
        self._test_y = tensors['test']['y']

        self._train_time = train_time
        self._batch_size = batch_size

    def set_tensors(self, train_X, train_y, valid_X, valid_y, test_X, test_y):
        self._train_X = train_X
        self._train_y = train_y
        self._valid_X = valid_X
        self._valid_y = valid_y
        self._test_X = test_X
        self._test_y = test_y

    def train(self):
        check_pointer = ModelCheckpoint(filepath=self._weights,
                                        verbose=1, save_best_only=True)
        if self._load_weights:
            if check_path_exists(self._weights):
                print('Loading preserved weights before training.')
                self._model.load_weights(self._weights)

        start_time = time.time()

        while True:
            self._model.fit(self._train_X, self._train_y,
                            validation_data=(self._valid_X, self._valid_y),
                            epochs=1, batch_size=self._batch_size, callbacks=[check_pointer], verbose=2)

            current_time = time.time()
            elapsed_time = int((current_time - start_time) / 60)
            print('Elapsed time: {0} minutes'.format(elapsed_time))
            if elapsed_time >= self._train_time:
                break


class MultiTrainer:
    def __init__(self, trainers):
        self._trainers = trainers

    def train(self):
        for t in self._trainers:
            t.train()


class GANTrainer:
    """
    GAN training class. It plots the result of the training on a specified period of training epochs.
    """
    def __init__(self, models, optimizers, losses, loss_weights, weights, load_weights, tensors, compare_path='.',
                 **kwargs):
        """
        Initialization function for the GANTrainer.

        :param models:
        :param optimizers:
        :param losses:
        :param loss_weights:
        :param weights:
        :param load_weights:
        :param tensors:
        """
        self._generator = models['generator']
        self._discriminator = models['discriminator']
        self._generator_weights = weights['generator']
        self._discriminator_weights = weights['discriminator']
        self._load_weights = load_weights

        self._train_X = tensors['train']['X']
        self._train_y = tensors['train']['y']
        self._valid_X = tensors['valid']['X']
        self._valid_y = tensors['valid']['y']
        self._test_X = tensors['test']['X']
        self._test_y = tensors['test']['y']

        self._optimizer_g = optimizers['generator']
        self._optimizer_d = optimizers['discriminator']
        self._optimizer_gan = optimizers['gan']

        self._loss_g = losses['generator']
        self._loss_d = losses['discriminator']
        self._loss_gan = losses['gan']

        self._compare_path = compare_path

        self._compile_generator()
        self._compile_discriminator()

        self._gan = self.create_gan()

    def _compile_generator(self):
        self._generator.compile(optimizer=self._optimizer_g, loss=self._loss_g, metrics=['accuracy'])

    def _compile_discriminator(self):
        self._discriminator.compile(optimizer=self._optimizer_d, loss=self._loss_d, metrics=['accuracy'])

    def plot_generated_images(self, epoch, examples=10, dim=(2, 5), figsize=(15, 10), base_path='.'):
        images = self._test_X[np.random.randint(low=0, high=self._test_X.shape[0], size=examples)]
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

        indexes = np.random.randint(low=0, high=self._test_X.shape[0], size=examples)
        images = self._test_X[indexes]
        generated_images = self._generator.predict(images)
        real_images = self._test_y[indexes]

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

        inputs = Input(self._train_X.shape[1:])
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
            print("Epoch %d" % e)
            for _ in tqdm(range(batch_size)):
                # generate random noise as an input to initialize the generator
                # noise_numbers = np.random.randint(1, self._train_X.shape[:1], batch_size)
                # noise = np.random.normal(0,1, [batch_size, 100])
                noise = self._train_X[np.random.randint(low=0, high=self._train_X.shape[0], size=batch_size)]

                generated_images = self._generator.predict(noise)

                # Get a random set of  real images
                image_batch = self._train_y[np.random.randint(low=0, high=self._train_X.shape[0], size=batch_size)]

                # Construct different batches of real and fake data
                X = np.concatenate([image_batch, generated_images])

                # Labels for generated and real data
                y_dis = np.zeros(2 * batch_size)
                y_dis[:batch_size] = 0.9

                # Pre train discriminator on  fake and real data  before starting the gan.
                self._discriminator.trainable = True
                # self._compile_discriminator()
                self._discriminator.train_on_batch(X, y_dis)

                # Tricking the noised input of the Generator as real data
                noise = self._train_X[np.random.randint(low=0, high=self._train_X.shape[0], size=batch_size)]
                y_gen = np.ones(batch_size)

                # During the training of gan,
                # the weights of discriminator should be fixed.
                # We can enforce that by setting the trainable flag
                self._discriminator.trainable = False
                # self._compile_discriminator()

                # training  the GAN by alternating the training of the Discriminator
                # and training the chained GAN model with Discriminator’s weights freezed.
                self._gan.train_on_batch(noise, y_gen)

            if e == 1 or e % epochs_between_plots == 0:
                self.plot_images_for_compare(epoch=e, base_path=self._compare_path)

            if epochs_between_saves > 0 and e % epochs_between_saves == 0:
                self.save_models_weights()

            current_time = time.time()
            if ((current_time - start_time)/60) > max_train_time:
                print('Model {} has been trained for the max_train_time ({})'.format('NAME', max_train_time))
                break


class GANGridTrainer:
    def __init__(self, param_grid, weights_path, load_weights, tensors, compare_path, **kwargs):
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
        self._tensors = tensors
        self._compare_path = compare_path

    # How to save the weights for the grid?
    # May be keep a map how to

    # Go through all of the combinations
    def train(self):
        for m in self._param_grid['models']:
            for o in self._param_grid['optimizers']:
                for l in self._param_grid['losses']:
                    # Generate full paths for the weights here
                    t = GANTrainer(models=m, optimizers=o, losses=l, loss_weights=None,
                                   weights=None, load_weights=self._load_weights, tensors=tensors)
                    t.train(1000)


# loss_function
# optimizer
# weights - [ G, D]
# load_weights
# train_times - [G, D]

# We need to pass the training, validation and test data
tensors = {'train': {'X': None, 'y': None},
           'valid': {'X': None, 'y': None},
           'test': {'X': None, 'y': None}}

# Models that we want to use. The idea to pass the models as well is so we cantrain with different initializerz
models = {'generator': None, 'discriminator': None}

# Paths to the weights
weights = {'generator': '', 'discriminator': ''}

# Training time in hours
train_times = {'generator': '', 'discriminator': ''}

arguments = {'kernel_initializer': 'glorot_uniform', 'loss_function': '',
             'optimizer': 'adam', 'weights': weights, 'load_weights': True,
             'train_times': train_times}


def create_tensors_dict(train_X, train_y, valid_X, valid_y, test_X, test_y):
    tensors = {'train': {'X': train_X, 'y': train_y},
               'valid': {'X': valid_X, 'y': valid_y},
               'test': {'X': test_X, 'y': test_y}}

    return tensors


def trainer(model, weights, load_weights, tensors, train_time=1, batch_size=32):
    check_pointer = ModelCheckpoint(filepath=weights,
                                    verbose=1, save_best_only=True)

    train_X = tensors['train']['X']
    train_y = tensors['train']['y']
    valid_X = tensors['valid']['X']
    valid_y = tensors['valid']['y']
    test_X = tensors['test']['X']
    test_X = tensors['test']['y']

    if load_weights:
        if check_path_exists(weights):
            print('Loading preserved weights before training.')
            model.load_weights(weights)

    start_time = time.time()

    while True:
        model.fit(train_X, train_y,
                  validation_data=(valid_X, valid_y),
                  epochs=1, batch_size=batch_size, callbacks=[check_pointer], verbose=2)

        current_time = time.time()
        elapsed_time = int((current_time - start_time) / 60)
        print('Elapsed time: {0} minutes'.format(elapsed_time))
        if elapsed_time >= train_time:
            break


def gan_trainer(models, weights, load_weights, tensors, train_times):
    # if models['generator']
    pass


def multi_trainer(dataset, trainers_data):
    if dataset is None:
        return

    if trainers_data is None:
        return

    for t_data in trainers_data:
        gan_trainer(tensors, t_data)
