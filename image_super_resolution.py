import logging
from glob import glob
from models import generator_with_residual, discriminator
from data_loader import load_images_with_truth
from loss_functions import perceptual_plus_texture_loss
from keras.optimizers import Adam
from keras.losses import mean_squared_error, binary_crossentropy
from trainer import GANTrainer
from trainer import DataType, FilesDataProvider, InMemoryDataProvider


# Create and configure logger
logger = logging.getLogger('image_super_resolution')
logger.setLevel(logging.DEBUG)

# Create file handler for the logs
file_handler = logging.FileHandler('./logs/image_super_resolution.log')
file_handler.setLevel(logging.DEBUG)

# Create console handler for the logs
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# Training parameters
# Use on the fly file loader instead of loading everything in memory first
use_file_loader = True
train_epochs = 100000
train_batch_size = 32
epochs_between_plots = 50
epochs_between_saves = 50
load_saved_checkpoint_before_training = False
max_train_time = 60 * 24    # 24 hours
optimizer_learning_rate = 0.0002

# Define the dataset path
train_dataset_path = './data/MSCOCO/train/*'
validation_dataset_path = './data/MSCOCO/valid_small/*'
test_dataset_path = './data/MSCOCO/test_small/*'

if use_file_loader:
    training_files = glob(train_dataset_path)
    training_provider = FilesDataProvider(training_files, train_batch_size)

    test_files = glob(test_dataset_path)
    test_provider = FilesDataProvider(test_files, train_batch_size)

    data_providers = {DataType.Training: training_provider, DataType.Test: test_provider}
else:
    logger.info('Loading train data:')
    train_X, train_y = load_images_with_truth(train_dataset_path, 4, normalize=True)
    logger.info('Loading validation data:')
    validation_X, validation_y = load_images_with_truth(validation_dataset_path, 4, normalize=True)
    logger.info('Loading test data:')
    test_X, test_y = load_images_with_truth(test_dataset_path, 4, normalize=True)

    logger.info("Train images: {}".format(len(train_X)))
    logger.info('Validation images: {}'.format(len(validation_X)))
    logger.info("Test images: {}".format(len(test_X)))

    training_provider = InMemoryDataProvider(X=train_X, y=train_y, batch_size=train_batch_size)
    test_provider = InMemoryDataProvider(X=test_X, y=test_y, batch_size=train_batch_size)

    data_providers = {DataType.Training: training_provider, DataType.Test: test_provider}

X, y = data_providers[DataType.Training].get_batch(1)
generator = generator_with_residual(input_shape=X.shape[1:], summary=True)
discriminator = discriminator(input_shape=y.shape[1:], summary=True)

optimizer = Adam(lr=optimizer_learning_rate)
models = {'generator': generator, 'discriminator': discriminator}
optimizers = {'generator': optimizer, 'discriminator': optimizer, 'gan': optimizer}
losses = {'generator': perceptual_plus_texture_loss, 'discriminator': binary_crossentropy, 'gan': binary_crossentropy}
weights = {'generator': './saved_models/weights.generator.hdf5',
           'discriminator': './saved_models/weights.discriminator.hdf5'}

logger.info('Create GANTrainer.')
trainer = GANTrainer(models=models, optimizers=optimizers, losses=losses, loss_weights=None, weights=weights,
                     load_weights=load_saved_checkpoint_before_training, data_providers=data_providers,
                     compare_path='./compare/')
logger.info('Start training.')
trainer.train(epochs=train_epochs, batch_size=train_batch_size, epochs_between_plots=epochs_between_plots,
              epochs_between_saves=epochs_between_saves, max_train_time=max_train_time)

logger.info('All done!')

