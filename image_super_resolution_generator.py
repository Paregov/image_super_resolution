import sys
import gc
from glob import glob
from models import generator_no_residual, generator_with_residual
from utils import check_path_exists
from data_loader import load_images_with_truth
from data_generator import DataGenerator
from loss_functions import perceptual_loss, perceptual_loss_16, perceptual_loss_19
from loss_functions import texture_loss_multi_layers, perceptual_plus_texture_loss, perceptual_16_plus_texture_loss
from visualizations import plot_images_for_compare_separate, compare_models_single_image
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

# Enable this flag if running on the laptop with smaller GPU.
running_on_laptop = False

# This will use the smaller datasets (train_small, val_small, test_small).
use_small_dataset = True

# Set to true if you want to use CelebA dataset. Otherwise it will use MS COCO.
use_dataset_celeba = False

# Set to true and it will not execute training. Usefull when just want to plot the results.
disable_training = False

# What to be the verbose level during training.
training_verbose = 2

enable_p = False
enable_p16 = False
enable_p19 = False
enable_t = False
enable_pt = False
enable_pt16 = False
enable_pt16_bci = True
enable_pt_bci = True
enable_pt16_no_res = True

train_epochs = 10
train_batch_size = 32
test_image_index_to_show = range(20)
optimizer = Adam(lr=0.0001)

if running_on_laptop:
    train_epochs = 100
    train_batch_size = 8
    test_image_index_to_show = range(20)
    optimizer = Adam(lr=0.001)


# Define the dataset path
dataset = "MSCOCO"
if use_dataset_celeba:
    dataset = "celeba"

if not use_small_dataset:
    train_dataset_path = './data/{0}/train/*'.format(dataset)
    validation_dataset_path = './data/{0}/val/*'.format(dataset)
    test_dataset_path = './data/{0}/test/*'.format(dataset)
else:
    train_dataset_path = './data/{0}/train_small/*'.format(dataset)
    validation_dataset_path = './data/{0}/val_small/*'.format(dataset)
    test_dataset_path = './data/{0}/test_small/*'.format(dataset)

print(train_dataset_path)
print(validation_dataset_path)
print(test_dataset_path)

perceptual_loss_checkpint_path = './saved_models/weights.best.train.mscoco.pl.hdf5'
perceptual_loss_16_checkpoint_path = './saved_models/weights.best.train.mscoco.pl16.hdf5'
perceptual_loss_19_checkpoint_path = './saved_models/weights.best.train.mscoco.pl19.hdf5'
texture_loss_ml_checkpoint_path = './saved_models/weights.best.train.mscoco.tl_ml.hdf5'
texture_plus_perceptual_loss_checkpoint_path = './saved_models/weights.best.train.mscoco.tl_plus_pl.hdf5'
perceptual_16_plus_texture_loss_checkpoint_path = './saved_models/weights.best.train.mscoco.pl16_plus_tl.hdf5'
perceptual_16_plus_texture_loss_bci_checkpoint_path = './saved_models/weights.best.train.mscoco.pl16_plus_tl_bci.hdf5'
perceptual_plus_texture_loss_bci_checkpoint_path = './saved_models/weights.best.train.mscoco.pl_plus_tl_bci.hdf5'
perceptual_16_plus_texture_loss_no_res_checkpoint_path = './saved_models/weights.best.train.mscoco.pl16_plus_tl_no_res.hdf5'

if use_dataset_celeba:
    perceptual_loss_checkpint_path = './saved_models/weights.best.train.celeba.pl.hdf5'
    perceptual_loss_16_checkpoint_path = './saved_models/weights.best.train.celeba.pl16.hdf5'
    perceptual_loss_19_checkpoint_path = './saved_models/weights.best.train.celeba.pl19.hdf5'
    texture_loss_ml_checkpoint_path = './saved_models/weights.best.train.celeba.tl_ml.hdf5'
    texture_plus_perceptual_loss_checkpoint_path = './saved_models/weights.best.train.celeba.tl_plus_pl.hdf5'
    perceptual_16_plus_texture_loss_checkpoint_path = './saved_models/weights.best.train.celeba.pl16_plus_tl.hdf5'
    perceptual_16_plus_texture_loss_bci_checkpoint_path = './saved_models/weights.best.train.celeba.pl16_plus_tl_bci.hdf5'
    perceptual_plus_texture_loss_bci_checkpoint_path = './saved_models/weights.best.train.celeba.pl_plus_tl_bci.hdf5'
    perceptual_16_plus_texture_loss_no_res_checkpoint_path = './saved_models/weights.best.train.celeba.pl16_plus_tl_no_res.hdf5'


print('Loading train data:')
train_filenames = glob(train_dataset_path)
print('Loading validation data:')
validation_filenames = glob(validation_dataset_path)
print('Loading test data:')
test_filenames = glob(test_dataset_path)
test_data, test_truth = load_images_with_truth(test_dataset_path, 4)

training_samples = len(train_filenames)
validation_samples = len(validation_filenames)
test_samples = len(test_filenames)

print("Train images: ", training_samples)
print('Validation images: ', validation_samples)
print("Test images: ", test_samples)

test_data_tensors = test_data.astype('float32')/255
test_truth_tensors = test_truth.astype('float32')/255

train_data_shape = (32,32,3)


# NOTE: Some of the parameters are used from the global space
def model_train(model, optimizer, loss_function, checkpoint_path, verbose=2):
    try:
        training_generator = DataGenerator(image_filenames=train_filenames, batch_size=train_batch_size)
        validation_generator = DataGenerator(image_filenames=validation_filenames, batch_size=train_batch_size)

        model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath=checkpoint_path,
                                       verbose=verbose, save_best_only=True)
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=verbose)
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


if enable_p:
    model = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    if not disable_training:
        print("Training P")
        model_train(model=model, optimizer=optimizer, loss_function=perceptual_loss,
                    checkpoint_path=perceptual_loss_checkpint_path, verbose=training_verbose)
    model_predict(model, perceptual_loss_checkpint_path)
    del model
    gc.collect()

if enable_p19:
    model19 = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    if not disable_training:
        print("Training P19")
        model_train(model=model19, optimizer=optimizer, loss_function=perceptual_loss_19,
                    checkpoint_path=perceptual_loss_19_checkpoint_path)
    model_predict(model=model19, checkpoint_path=perceptual_loss_19_checkpoint_path)
    del model19
    gc.collect()

if enable_p16:
    model16 = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    if not disable_training:
        print("Training P16")
        model_train(model=model16, optimizer=optimizer, loss_function=perceptual_loss_16,
                    checkpoint_path=perceptual_loss_16_checkpoint_path, verbose=training_verbose)
    model_predict(model=model16, checkpoint_path=perceptual_loss_16_checkpoint_path)
    del model16
    gc.collect()

if enable_t:
    model_tl_ml = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    if not disable_training:
        print("Training T")
        model_train(model=model_tl_ml, optimizer=optimizer, loss_function=texture_loss_multi_layers,
                    checkpoint_path=texture_loss_ml_checkpoint_path, verbose=training_verbose)
    model_predict(model=model_tl_ml, checkpoint_path=texture_loss_ml_checkpoint_path)
    del model_tl_ml
    gc.collect()

if enable_pt:
    model_tl_plus_pl = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    if not disable_training:
        print("Training PT")
        model_train(model=model_tl_plus_pl, optimizer=optimizer, loss_function=perceptual_plus_texture_loss,
                    checkpoint_path=texture_plus_perceptual_loss_checkpoint_path, verbose=training_verbose)
    model_predict(model=model_tl_plus_pl, checkpoint_path=texture_plus_perceptual_loss_checkpoint_path)
    del model_tl_plus_pl
    gc.collect()

if enable_pt16:
    model_pl_16_plus_tl = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    if not disable_training:
        print("Training P16")
        model_train(model=model_pl_16_plus_tl, optimizer=optimizer, loss_function=perceptual_16_plus_texture_loss,
                    checkpoint_path=perceptual_16_plus_texture_loss_checkpoint_path, verbose=training_verbose)
    model_predict(model=model_pl_16_plus_tl, checkpoint_path=perceptual_16_plus_texture_loss_checkpoint_path)
    del model_pl_16_plus_tl
    gc.collect()

if enable_pt16_bci:
    model_pl_16_plus_tl_bci = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=True)
    if not disable_training:
        print("Training PT16_BCI")
        model_train(model=model_pl_16_plus_tl_bci, optimizer=optimizer, loss_function=perceptual_16_plus_texture_loss,
                    checkpoint_path=perceptual_16_plus_texture_loss_bci_checkpoint_path, verbose=training_verbose)
    model_predict(model=model_pl_16_plus_tl_bci, checkpoint_path=perceptual_16_plus_texture_loss_bci_checkpoint_path)
    del model_pl_16_plus_tl_bci
    gc.collect()

if enable_pt_bci:
    model_pl_plus_tl_bci = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=True)
    if not disable_training:
        print("Training PT_BCI")
        model_train(model=model_pl_plus_tl_bci, optimizer=optimizer, loss_function=perceptual_plus_texture_loss,
                    checkpoint_path=perceptual_plus_texture_loss_bci_checkpoint_path, verbose=training_verbose)
    model_predict(model=model_pl_plus_tl_bci, checkpoint_path=perceptual_plus_texture_loss_bci_checkpoint_path)
    del model_pl_plus_tl_bci
    gc.collect()

if enable_pt16_no_res:
    model_pl_16_plus_tl_no_res = generator_no_residual(input_shape=train_data_shape, summary=False)
    if not disable_training:
        print("Training PT16_NO_RES")
        model_train(model=model_pl_16_plus_tl_no_res, optimizer=optimizer,
                    loss_function=perceptual_16_plus_texture_loss,
                    checkpoint_path=perceptual_16_plus_texture_loss_no_res_checkpoint_path, verbose=training_verbose)
    model_predict(model=model_pl_16_plus_tl_no_res,
                  checkpoint_path=perceptual_16_plus_texture_loss_no_res_checkpoint_path)
    del model_pl_16_plus_tl_no_res
    gc.collect()

models_data = []

if enable_p:
    model = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    models_data.append({'name': "P", 'model': model, 'checkpoint': perceptual_loss_checkpint_path})

if enable_p16:
    model19 = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    models_data.append({'name': "P VGG19", 'model': model19, 'checkpoint': perceptual_loss_19_checkpoint_path})

if enable_p19:
    model16 = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    models_data.append({'name': "P VGG16", 'model': model16, 'checkpoint': perceptual_loss_16_checkpoint_path})

if enable_t:
    model_tl_ml = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    models_data.append({'name': "T", 'model': model_tl_ml, 'checkpoint': texture_loss_ml_checkpoint_path})

if enable_pt:
    model_tl_plus_pl = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    models_data.append({'name': "PT VGG19", 'model': model_tl_plus_pl,
                        'checkpoint': texture_plus_perceptual_loss_checkpoint_path})

if enable_pt16:
    model_pl_16_plus_tl = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=False)
    models_data.append({'name': "PT VGG16", 'model': model_pl_16_plus_tl,
                        'checkpoint': perceptual_16_plus_texture_loss_checkpoint_path})

if enable_pt16_bci:
    model_pl_16_plus_tl_bci = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=True)
    models_data.append({'name': "PT VGG16 (BCI)", 'model': model_pl_16_plus_tl_bci,
                        'checkpoint': perceptual_16_plus_texture_loss_bci_checkpoint_path})

if enable_pt_bci:
    model_pl_plus_tl_bci = generator_with_residual(input_shape=train_data_shape, summary=False, add_bicubic=True)
    models_data.append({'name': "PT VGG19 (BCI)", 'model': model_pl_plus_tl_bci,
                        'checkpoint': perceptual_plus_texture_loss_bci_checkpoint_path})

if enable_pt16_no_res:
    model_pl_16_plus_tl_no_res = generator_no_residual(input_shape=train_data_shape, summary=False)
    models_data.append({'name': "PT VGG16 (NR)", 'model': model_pl_16_plus_tl_no_res,
                        'checkpoint': perceptual_16_plus_texture_loss_no_res_checkpoint_path})

compare_models_single_image(test_data_tensors, test_truth_tensors, models_data, test_image_index_to_show,
                            show_input=True)

