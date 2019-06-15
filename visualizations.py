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


# TODO: Refactor this function so it accepts predictions as array of dictionaries
#  [{'image': image, 'name': 'name'}, ...] and remove the second array 'model_names'
def plot_multi_model_images_for_compare_separate(inputs, predictions, model_names, truths, indexes, show_input=False, show_interpolated=True):
    length = len(indexes)
    predictions_count = len(predictions)

    width = predictions_count + 1
    if show_input:
        width += 1
        
    if show_interpolated:
        width += 1
    
    for i in range(length):
        pos = 0
        f, axarr = plt.subplots(1, width, figsize=(25, 10))
        img_index = indexes[i]
        if show_input:
            axarr[pos].imshow(inputs[img_index])
            axarr[pos].set_title('input')
            pos += 1
        
        if show_input:
            axarr[pos].imshow(inputs[img_index], interpolation='bicubic')
            axarr[pos].set_title('bicubic')
            pos += 1

        index = 0
        for prediction in predictions:
            axarr[pos].imshow((prediction[img_index]*255).astype(np.uint8))
            axarr[pos].set_title(model_names[index])
            pos += 1
            index += 1

        axarr[pos].imshow((truths[img_index]*255).astype(np.uint8))
        axarr[pos].set_title('truth')
        plt.show()


def plot_single_image_multi_model(input_image, predictions, truth, show_input=True, show_interpolated=True,
                                  figsize=(25, 5)):
    """
    Plot a single image that is predicted with multiple models for compare.
    We are going to show 4 images per row and will create as many rows as needed to show all of the predictions,
    input, interpolated and truth
    
    Params:
    input_image - the input image that is being enhanced
    predictions - list of maps [{'image': image, 'name': 'name'}, ...]
    """
    images_per_row = 4
    predictions_count = len(predictions)

    total_count = predictions_count + 1
    if show_input:
        total_count += 1
        
    if show_interpolated:
        total_count += 1
    
    rows = int(total_count / images_per_row)
    if total_count % images_per_row != 0:
        rows += 1
    
    prediction_index = 0
    for i in range(rows):
        pos = 0
        f, axarr = plt.subplots(1, images_per_row, figsize=figsize)

        if i == 0 and show_input:
            axarr[pos].imshow(input_image)
            axarr[pos].set_title('input')
            pos += 1
        
        if i == 0 and show_interpolated:
            axarr[pos].imshow(input_image, interpolation='bicubic')
            axarr[pos].set_title('bicubic')
            pos += 1

        if i == 0:
            axarr[pos].imshow((truth*255).astype(np.uint8))
            axarr[pos].set_title('truth')
            pos += 1

        for p in range(predictions_count):
            prediction = predictions[prediction_index]
            axarr[pos].imshow((prediction['image']*255).astype(np.uint8))
            axarr[pos].set_title(prediction['name'])
            pos += 1
            prediction_index += 1
            
            if pos >= images_per_row or prediction_index >= predictions_count:
                break
            
        plt.show()


def compare_models(test_data_tensors, test_truth_tensors, models_data, test_image_index_to_show, show_input=True,
                   show_interpolated=True):
    predictions = []
    model_names = []
    for model_data in models_data:
        model_names.append(model_data['name'])
        model_data['model'].load_weights(model_data['checkpoint'])
        predictions.append(model_data['model'].predict(test_data_tensors))
    
    print('Plotting the results...')
    plot_multi_model_images_for_compare_separate(test_data_tensors, predictions,
                                                 model_names, test_truth_tensors,
                                                 test_image_index_to_show,
                                                 show_input=show_input,
                                                 show_interpolated=show_interpolated)

    print('All done!')

    
def compare_models_single_image(test_data_tensors, test_truth_tensors, models_data, test_image_index_to_show,
                                show_input=True, show_interpolated=True, figsize=(25,5)):
    predictions_per_model = []
    for model_data in models_data:
        model_data['model'].load_weights(model_data['checkpoint'])
        predictions_per_model.append({'data': model_data['model'].predict(test_data_tensors),
                                      'name': model_data['name']})
    
    print('Plotting the results...')
    for i in test_image_index_to_show:
        predictions = []
        for pred in predictions_per_model:
            predictions.append({'image': pred['data'][i], 'name': pred['name']})

        print("Image: {0}".format(i + 1))
        plot_single_image_multi_model(test_data_tensors[i], predictions, test_truth_tensors[i],
                                      show_input=show_input,
                                      show_interpolated=show_interpolated,
                                      figsize=figsize)
        print("")

    print('All done!')
