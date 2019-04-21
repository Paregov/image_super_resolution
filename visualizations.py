import numpy as np
import matplotlib.pyplot as plt


def plot_images_for_compare(inputs, predictions, truths, indexes):
    length = len(indexes)
    f, axarr = plt.subplots(length, 3, figsize=(10, 10))
    for i in range(length):
        img_index = indexes[i]
        axarr[i,0].imshow(inputs[img_index])
        axarr[i,1].imshow((predictions[img_index]*255).astype(np.uint8))
        axarr[i,2].imshow((truths[img_index]*255).astype(np.uint8))
    plt.show()


def plot_images_for_compare_separate(inputs, predictions, truths, indexes):
    length = len(indexes)
    for i in range(length):
        f, axarr = plt.subplots(1, 3, figsize=(10, 5))
        img_index = indexes[i]
        axarr[0].imshow(inputs[img_index])
        axarr[1].imshow((predictions[img_index]*255).astype(np.uint8))
        axarr[2].imshow((truths[img_index]*255).astype(np.uint8))
        plt.show()
