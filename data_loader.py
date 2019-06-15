import numpy as np
from glob import glob
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


def load_image_with_truth(file_name, scale=4, min_side=384, downsample_size=256, hr_size=128):
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

    # Downsample to the passed downsample_size
    img = img.resize((downsample_size, downsample_size), Image.ANTIALIAS)

    # Crop the image to the high resolution dimensions requested
    img = crop_center_rectangle(img, hr_size)

    # Now is time to do the scaling to low resolution
    low_resolution = img.resize((hr_size // scale, hr_size // scale), Image.ANTIALIAS)
    x = img_to_array(low_resolution)
    y = img_to_array(img)

    return np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)


def load_images_with_truth(folder_path, scale=4, min_side=384, downsample_size=256, hr_size=128):
    X = []
    y = []
    for img_path in tqdm(glob(folder_path)):
        temp_X, temp_y = load_image_with_truth(img_path, scale, min_side, downsample_size, hr_size)
        if temp_X is not None:
            X.append(temp_X)
            y.append(temp_y)

    return np.vstack(X), np.vstack(y)


def load_images_list_with_truth(images_list, scale=4, min_side=384, downsample_size=256, hr_size=128, normalize=False):
    X = []
    y = []
    for img_path in images_list:
        temp_X, temp_y = load_image_with_truth(img_path, scale, min_side, downsample_size, hr_size)
        if temp_X is not None:
            X.append(temp_X)
            y.append(temp_y)

    if normalize:
        return (np.vstack(X).astype('float32') / 255), (np.vstack(y).astype('float32') / 255)

    return np.vstack(X), np.vstack(y)


###############################################################################################
# Some unused functions
###############################################################################################
# TODO: Implement a function that loads together so it's faster
# Function from the original research
def load_image(file_name, scale=4, min_side_in_pixels=384, downsample_side=256):
    try:
        img = load_img(file_name)
    except IOError:
        return None

    width, height = img.size
    if width < min_side_in_pixels or height < min_side_in_pixels:
        return None

    # Let's get the center part of the image
    # We are going to make it square with smaller side used as a base
    side = min(width, height)
    center_width = width / 2
    center_height = height / 2
    half = min_side_in_pixels / 2

    img = img.crop((center_width - half, center_height - half, floor(center_width + half), floor(center_height + half)))

    # Now is time to do the scaling
    img = img.resize((min_side_in_pixels // scale, min_side_in_pixels // scale), Image.ANTIALIAS)
    x = img_to_array(img)

    return np.expand_dims(x, axis=0)


def load_images(folder_path, scale=4, min_side_in_pixels=256):
    list_of_tensors = []
    for img_path in tqdm(glob(folder_path)):
        temp_image = load_image(img_path, scale, min_side_in_pixels)
        if temp_image is not None:
            list_of_tensors.append(temp_image)

    return np.vstack(list_of_tensors)

