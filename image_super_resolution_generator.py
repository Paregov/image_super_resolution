import sys
import gc
from glob import glob
from enum import Enum
from models import generator_no_residual, generator_with_residual
from utils import check_path_exists
from data_loader import load_images_with_truth
from data_generator import DataGenerator
from loss_functions import perceptual_loss, perceptual_loss_16, perceptual_loss_19
from loss_functions import texture_loss_multi_layers, perceptual_plus_texture_loss, perceptual_16_plus_texture_loss
from visualizations import plot_images_for_compare_separate, compare_models_single_image
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam


class DataSet(Enum):
    MSCOCO = 1
    CELEBA = 2


class DataSetSize(Enum):
    SMALL = 1
    MEDIUM = 2
    FULL = 3


# Enable this flag if running on the laptop with smaller GPU.
running_on_laptop = False

# Set to true if you want to use CelebA dataset. Otherwise it will use MS COCO.
dataset = DataSet.MSCOCO

# This will use the smaller datasets (train_small, val_small, test_small).
dataset_size = DataSetSize.SMALL

# Set to true and it will not execute training. Usefull when just want to plot the results.
disable_training = False

# Set to true if you want to load the current weights from the folder before training, so you can
# continue with the training
load_weights_before_training = False

use_generator = False

# What to be the verbose level during training.
training_verbose = 2

enable_p = False
enable_p16 = False
enable_p19 = False
enable_t = False
enable_pt = False
enable_pt16 = False
enable_pt16_bci = False
enable_pt_bci = True
enable_pt16_no_res = False

train_epochs = 100
train_batch_size = 48
test_image_index_to_show = range(2)
optimizer = Adam(lr=0.0005)

if running_on_laptop:
    train_epochs = 100
    train_batch_size = 8
    test_image_index_to_show = range(20)
    optimizer = Adam(lr=0.001)

# Define the dataset path
if dataset == DataSet.MSCOCO:
    dataset_name = "MSCOCO"
elif dataset == DataSet.CELEBA:
    dataset_name = "celeba"

if dataset_size == DataSetSize.FULL:
    train_dataset_path = './data/{0}/train/*'.format(dataset_name)
    validation_dataset_path = './data/{0}/val/*'.format(dataset_name)
    test_dataset_path = './data/{0}/test/*'.format(dataset_name)
elif dataset_size == DataSetSize.MEDIUM:
    train_dataset_path = './data/{0}/train_middle/*'.format(dataset_name)
    validation_dataset_path = './data/{0}/val_middle/*'.format(dataset_name)
    test_dataset_path = './data/{0}/test_small/*'.format(dataset_name)
elif dataset_size == DataSetSize.SMALL:
    train_dataset_path = './data/{0}/train_small/*'.format(dataset_name)
    validation_dataset_path = './data/{0}/val_small/*'.format(dataset_name)
    test_dataset_path = './data/{0}/test_small/*'.format(dataset_name)

print(train_dataset_path)
print(validation_dataset_path)
print(test_dataset_path)

if dataset == DataSet.MSCOCO:
    checkpoint_path_p = './saved_models/weights.best.train.mscoco.p.hdf5'
    checkpoint_path_p16 = './saved_models/weights.best.train.mscoco.p16.hdf5'
    checkpoint_path_p19 = './saved_models/weights.best.train.mscoco.p19.hdf5'
    checkpoint_path_t = './saved_models/weights.best.train.mscoco.t.hdf5'
    checkpoint_path_pt = './saved_models/weights.best.train.mscoco.pt.hdf5'
    checkpoint_path_pt16 = './saved_models/weights.best.train.mscoco.pt16.hdf5'
    checkpoint_path_pt16_bci = './saved_models/weights.best.train.mscoco.pt16_bci.hdf5'
    checkpoint_path_pt_bci = './saved_models/weights.best.train.mscoco.pt_bci.hdf5'
    checkpoint_path_pt16_no_res = './saved_models/weights.best.train.mscoco.pt16_no_res.hdf5'
elif dataset == DataSet.CELEBA:
    checkpoint_path_p = './saved_models/weights.best.train.celeba.p.hdf5'
    checkpoint_path_p16 = './saved_models/weights.best.train.celeba.p16.hdf5'
    checkpoint_path_p19 = './saved_models/weights.best.train.celeba.p19.hdf5'
    checkpoint_path_t = './saved_models/weights.best.train.celeba.t.hdf5'
    checkpoint_path_pt = './saved_models/weights.best.train.celeba.pt.hdf5'
    checkpoint_path_pt16 = './saved_models/weights.best.train.celeba.pt16.hdf5'
    checkpoint_path_pt16_bci = './saved_models/weights.best.train.celeba.pt16_bci.hdf5'
    checkpoint_path_pt_bci = './saved_models/weights.best.train.celeba.pt_bci.hdf5'
    checkpoint_path_pt16_no_res = './saved_models/weights.best.train.celeba.pt16_no_res.hdf5'

if use_generator:
    print('Loading train data file names:')
    train_file_names = glob(train_dataset_path)
    training_samples = len(train_file_names)

    print('Loading validation data file names:')
    validation_file_names = glob(validation_dataset_path)
    validation_samples = len(validation_file_names)
else:
    print('Loading training data:')
    train_data_tensors, train_truth_tensors = load_images_with_truth(train_dataset_path, 4, normalize=True)
    training_samples = len(train_data_tensors)

    print('Loading validation data:')
    validation_data_tensors, validation_truth_tensors = load_images_with_truth(validation_dataset_path, 4,
                                                                               normalize=True)
    validation_samples = len(validation_data_tensors)

print("Training images: ", training_samples)
print('Validation images: ', validation_samples)

print('Loading test data:')
test_data_tensors, test_truth_tensors = load_images_with_truth(test_dataset_path, 4, normalize=True)

test_samples = len(test_data_tensors)
print("Test images: ", test_samples)

train_data_shape = (32, 32, 3)


# NOTE: Some of the parameters are used from the global space
def model_train(model, optimizer, loss_function, checkpoint_path, verbose=2):
    try:
        if use_generator:
            training_generator = DataGenerator(image_filenames=train_file_names, batch_size=train_batch_size)
            validation_generator = DataGenerator(image_filenames=validation_file_names, batch_size=train_batch_size)

        if load_weights_before_training:
            if check_path_exists(checkpoint_path):
                print("Loading checkpoint: ", checkpoint_path)
                model.load_weights(checkpoint_path)
            else:
                print("Checkpoint {0} doesn't exist. Skip loading.".format(checkpoint_path))

        model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath=checkpoint_path,
                                       verbose=verbose, save_best_only=True)
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=verbose)
        if use_generator:
            model.fit_generator(generator=training_generator,
                                steps_per_epoch=(training_samples // train_batch_size),
                                validation_data=validation_generator,
                                validation_steps=(validation_samples // train_batch_size),
                                epochs=train_epochs,
                                use_multiprocessing=False,      # When this is enabled it is failing
                                workers=1,
                                max_queue_size=32,
                                callbacks=[checkpointer, early_stopper],
                                verbose=verbose)
        else:
            model.fit(train_data_tensors, train_truth_tensors,
                      validation_data=(validation_data_tensors, validation_truth_tensors),
                      epochs=train_epochs,
                      batch_size=train_batch_size,
                      callbacks=[checkpointer, early_stopper],
                      verbose=verbose)
    except OSError:
        print("OSError error: ", sys.exc_info()[0])
    except:
        print("Unexpected error: ", sys.exc_info()[0])


# NOTE: Some of the parameters are used from the global space
def model_predict(model, checkpoint_path):
    return
    model.load_weights(checkpoint_path)

    print('Predicting...')
    predictions = model.predict(test_data_tensors)

    print('Plotting the results...')
    plot_images_for_compare_separate(test_data_tensors, predictions, test_truth_tensors, test_image_index_to_show)

    print('All done!')


# Here is the actual training of the models

if enable_p:
    model_p = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    if not disable_training:
        print("Training P")
        model_train(model=model_p, optimizer=optimizer, loss_function=perceptual_loss,
                    checkpoint_path=checkpoint_path_p, verbose=training_verbose)
    model_predict(model_p, checkpoint_path_p)
    del model_p
    gc.collect()

if enable_p19:
    model_p19 = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    if not disable_training:
        print("Training P19")
        model_train(model=model_p19, optimizer=optimizer, loss_function=perceptual_loss_19,
                    checkpoint_path=checkpoint_path_p19)
    model_predict(model=model_p19, checkpoint_path=checkpoint_path_p19)
    del model_p19
    gc.collect()

if enable_p16:
    model_p16 = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    if not disable_training:
        print("Training P16")
        model_train(model=model_p16, optimizer=optimizer, loss_function=perceptual_loss_16,
                    checkpoint_path=checkpoint_path_p16, verbose=training_verbose)
    model_predict(model=model_p16, checkpoint_path=checkpoint_path_p16)
    del model_p16
    gc.collect()

if enable_t:
    model_t = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    if not disable_training:
        print("Training T")
        model_train(model=model_t, optimizer=optimizer, loss_function=texture_loss_multi_layers,
                    checkpoint_path=checkpoint_path_t, verbose=training_verbose)
    model_predict(model=model_t, checkpoint_path=checkpoint_path_t)
    del model_t
    gc.collect()

if enable_pt:
    model_pt = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    if not disable_training:
        print("Training PT")
        model_train(model=model_pt, optimizer=optimizer, loss_function=perceptual_plus_texture_loss,
                    checkpoint_path=checkpoint_path_pt, verbose=training_verbose)
    model_predict(model=model_pt, checkpoint_path=checkpoint_path_pt)
    del model_pt
    gc.collect()

if enable_pt16:
    model_pt16 = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    if not disable_training:
        print("Training P16")
        model_train(model=model_pt16, optimizer=optimizer, loss_function=perceptual_16_plus_texture_loss,
                    checkpoint_path=checkpoint_path_pt16, verbose=training_verbose)
    model_predict(model=model_pt16, checkpoint_path=checkpoint_path_pt16)
    del model_pt16
    gc.collect()

if enable_pt16_bci:
    model_pt16_bci = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=True)
    if not disable_training:
        print("Training PT16_BCI")
        model_train(model=model_pt16_bci, optimizer=optimizer, loss_function=perceptual_16_plus_texture_loss,
                    checkpoint_path=checkpoint_path_pt16_bci, verbose=training_verbose)
    model_predict(model=model_pt16_bci, checkpoint_path=checkpoint_path_pt16_bci)
    del model_pt16_bci
    gc.collect()

if enable_pt_bci:
    model_pt_bci = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=True)
    if not disable_training:
        print("Training PT_BCI")
        model_train(model=model_pt_bci, optimizer=optimizer, loss_function=perceptual_plus_texture_loss,
                    checkpoint_path=checkpoint_path_pt_bci, verbose=training_verbose)
    model_predict(model=model_pt_bci, checkpoint_path=checkpoint_path_pt_bci)
    del model_pt_bci
    gc.collect()

if enable_pt16_no_res:
    model_pt16_no_res = generator_no_residual(input_shape=train_data_shape, summary=False)
    if not disable_training:
        print("Training PT16_NO_RES")
        model_train(model=model_pt16_no_res, optimizer=optimizer,
                    loss_function=perceptual_16_plus_texture_loss,
                    checkpoint_path=checkpoint_path_pt16_no_res, verbose=training_verbose)
    model_predict(model=model_pt16_no_res,
                  checkpoint_path=checkpoint_path_pt16_no_res)
    del model_pt16_no_res
    gc.collect()

models_data = []

if enable_p:
    model_p = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    models_data.append({'name': "P", 'model': model_p, 'checkpoint': checkpoint_path_p})

if enable_p19:
    model_p19 = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    models_data.append({'name': "P VGG19", 'model': model_p19, 'checkpoint': checkpoint_path_p19})

if enable_p16:
    model_p16 = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    models_data.append({'name': "P VGG16", 'model': model_p16, 'checkpoint': checkpoint_path_p16})

if enable_t:
    model_t = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    models_data.append({'name': "T", 'model': model_t, 'checkpoint': checkpoint_path_t})

if enable_pt:
    model_pt = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    models_data.append({'name': "PT", 'model': model_pt,
                        'checkpoint': checkpoint_path_pt})

if enable_pt16:
    model_pt16 = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    models_data.append({'name': "PT VGG16", 'model': model_pt16,
                        'checkpoint': checkpoint_path_pt16})

if enable_pt16_bci:
    model_pt16_bci = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=True)
    models_data.append({'name': "PT VGG16 (BCI)", 'model': model_pt16_bci,
                        'checkpoint': checkpoint_path_pt16_bci})

if enable_pt_bci:
    model_pt_bci = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=True)
    models_data.append({'name': "PT (BCI)", 'model': model_pt_bci,
                        'checkpoint': checkpoint_path_pt_bci})

if enable_pt16_no_res:
    model_pt16_no_res = generator_no_residual(input_shape=train_data_shape, summary=False)
    models_data.append({'name': "PT VGG16 (NR)", 'model': model_pt16_no_res,
                        'checkpoint': checkpoint_path_pt16_no_res})

compare_models_single_image(test_data_tensors, test_truth_tensors, models_data, test_image_index_to_show,
                            show_input=True, show_interpolated=False, figsize=(16, 4))

