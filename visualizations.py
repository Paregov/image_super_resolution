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


def plot_multi_model_images_for_compare_separate(inputs, predictions, truths, indexes, show_input=False):
    length = len(indexes)
    predictions_count = len(predictions)

    for i in range(length):
        pos = 0
        f, axarr = plt.subplots(1, predictions_count + 2, figsize=(25, 10))
        img_index = indexes[i]
        if show_input:
            axarr[pos].imshow(inputs[img_index])
            pos += 1

        for prediction in predictions:
            axarr[pos].imshow((prediction[img_index]*255).astype(np.uint8))
            pos += 1

        axarr[pos].imshow((truths[img_index]*255).astype(np.uint8))
        plt.show()


def compare_models(test_data_tensors, test_truth_tensors, models_data, test_image_index_to_show, show_input=False):
    models = []
    for model_data in models_data:
        model_data['model'].load_weights(model_data['checkpoint'])
        models.append(model_data['model'])
    
    predictions = []
    for model in models:
        predictions.append(model.predict(test_data_tensors))

    print('Plotting the results...')
    plot_multi_model_images_for_compare_separate(test_data_tensors, predictions, test_truth_tensors, test_image_index_to_show, show_input)

    print('All done!')
