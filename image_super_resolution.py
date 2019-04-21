from models import generator_no_residual, generator_with_residual, discriminator
from models import model_weights_no_residual, model_weights_with_residual
from data_loader import load_images_with_truth
from loss_functions import perceptual_loss, perceptual_loss_16, texture_loss, perceptual_plus_texture_loss
from keras.optimizers import Adam
from keras.losses import mean_squared_error, binary_crossentropy
from trainer import GANTrainer, create_tensors_dict, trainer

# Training parameters
train_epochs = 100000
train_batch_size = 32
epochs_between_plots = 100
epochs_between_saves = 100
load_saved_checkpoint_before_training = True
max_train_time = 60 * 24    # 24 hours

# Define the dataset path
train_dataset_path = './data/celeba/train_small/*'
validation_dataset_path = './data/celeba/valid_small/*'
test_dataset_path = './data/celeba/test_small/*'

print('Loading train data:')
train_data, train_truth = load_images_with_truth(train_dataset_path, 4)
print('Loading validation data:')
validation_data, validation_truth = load_images_with_truth(validation_dataset_path, 4)
print('Loading test data:')
test_data, test_truth = load_images_with_truth(test_dataset_path, 4)

print("Train images: ", len(train_data))
print('Validation images: ', len(validation_data))
print("Test images: ", len(test_data))

train_X = train_data.astype('float32')/255
train_y = train_truth.astype('float32')/255

validation_X = validation_data.astype('float32')/255
validation_y = validation_truth.astype('float32')/255

test_X = test_data.astype('float32')/255
test_y = test_truth.astype('float32')/255

tensors = create_tensors_dict(train_X, train_y, validation_X, validation_y, test_X, test_y)

generator = generator_with_residual(input_shape=train_data.shape[1:], summary=True)
discriminator = discriminator(input_shape=train_truth.shape[1:])

optimizer = Adam(lr=1e-4)
models = {'generator': generator, 'discriminator': discriminator}
optimizers = {'generator': optimizer, 'discriminator': optimizer, 'gan': optimizer}
losses = {'generator': perceptual_plus_texture_loss, 'discriminator': binary_crossentropy, 'gan': binary_crossentropy}
weights = {'generator': './saved_models/weights.generator.hdf5',
           'discriminator': './saved_models/weights.discriminator.hdf5'}

trainer = GANTrainer(models, optimizers, losses, None, weights, load_saved_checkpoint_before_training, tensors)
trainer.train(epochs=train_epochs, batch_size=train_batch_size, epochs_between_plots=epochs_between_plots,
              epochs_between_saves=epochs_between_saves, max_train_time=max_train_time)

print('All done!')

