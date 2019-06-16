import numpy as np
from PIL import Image


def save_image(image, file_path):
    result = Image.fromarray((image).astype(np.uint8))
    result.save(file_path)


def save_image_from_tensor(image, file_path, multiply=False):
    multiplier = 1
    if multiply:
        multiplier = np.uint8(255)

    result = Image.fromarray((image * multiplier).astype(np.uint8))
    result.save(file_path)

