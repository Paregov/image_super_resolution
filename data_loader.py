import numpy as np
from glob import glob
import ntpath
from tqdm import tqdm
from math import floor
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array


def crop_center_rectangle(img, side):
    width, height = img.size
    center_width = width / 2
    center_height = height / 2
    half = side / 2

    # Crop the image in the center so it's a rectangular
    return img.crop(
        (center_width - half, center_height - half, floor(center_width + half), floor(center_height + half)))


def load_image_with_truth(file_name, scale=4, min_side=384, downsample_size=256, hr_size=128, normalize=True):
    try:
        img = load_img(file_name)
    except IOError:
        return None, None

    width, height = img.size
    if width < min_side or height < min_side:
        return None, None

    # Let's get the center part of the image
    # We are going to make it square with smaller side used as a base
    img = crop_center_rectangle(img, min(width, height))

    # Down-sample to the passed downsample_size
    img = img.resize((downsample_size, downsample_size), Image.ANTIALIAS)

    # Crop the image to the high resolution dimensions requested
    img = crop_center_rectangle(img, hr_size)

    # Now is time to do the scaling to low resolution
    low_resolution = img.resize((hr_size // scale, hr_size // scale), Image.ANTIALIAS)
    x = img_to_array(low_resolution)
    y = img_to_array(img)

    if normalize:
        return (np.expand_dims(x, axis=0) / 255), (np.expand_dims(y, axis=0) / 255)

    return np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)


def load_images_with_truth(folder_path, scale=4, min_side=384, downsample_size=256, hr_size=128, normalize=False):
    X = []
    y = []
    for img_path in tqdm(glob(folder_path)):
        temp_X, temp_y = load_image_with_truth(img_path, scale, min_side, downsample_size, hr_size, normalize)
        if temp_X is not None:
            X.append(temp_X)
            y.append(temp_y)

    return np.vstack(X), np.vstack(y)


def load_images_list_with_truth(images_list, scale=4, min_side=384, downsample_size=256, hr_size=128, normalize=False):
    X = []
    y = []
    for img_path in images_list:
        temp_X, temp_y = load_image_with_truth(img_path, scale, min_side, downsample_size, hr_size, normalize)
        if temp_X is not None:
            X.append(temp_X)
            y.append(temp_y)

    return np.vstack(X), np.vstack(y)


def load_image_file(file_name, normalize=False, vstack=False):
    try:
        img = load_img(file_name)
    except IOError:
        return None, None

    img_array = img_to_array(img)

    result = np.expand_dims(img_array, axis=0)
    if normalize:
        result = result / 255
    
    if vstack:
        result = np.vstack(result)

    return result


def load_images_from_folder(folder_path, normalize=False):
    images = []
    for img_path in tqdm(glob(folder_path)):
        temp_image = load_image_file(img_path, normalize)
        if temp_image is not None:
            images.append(temp_image)

    return np.vstack(images)


def path_leaf(path):
    head, tail = ntpath.split(path)
    name = tail or ntpath.basename(head)
    return name.split('.')[0]


def load_images_from_folder_with_names(folder_path, normalize=False):
    result = []
    names = []
    for img_path in tqdm(glob(folder_path)):
        temp_image = load_image_file(img_path, normalize)
        if temp_image is not None:
            result.append({'image': np.vstack(temp_image), 'name': path_leaf(img_path)})

    return result

