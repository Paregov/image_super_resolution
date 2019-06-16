from data_loader import load_image_file
from data_saver import save_image_from_tensor
import matplotlib.pyplot as plt
import numpy as np
from visualizations import compare_images
from models import generator_no_residual, generator_with_residual, discriminator

class Predictor:
    def __init__(self, checkpoint_path, has_bicubic=False):
        self._checkpoint_path = checkpoint_path
        self._has_bicubic = has_bicubic

    def predict_from_file(self, image_path):
        img = load_image_file(file_name=image_path, normalize=True, vstack=True)
        model = generator_with_residual(input_shape=img.shape, summary=False, add_bicubic=self._has_bicubic)
        model.load_weights(self._checkpoint_path)

        to_predict = []
        to_predict.append(img)
        predictions = model.predict(np.array(to_predict))
        return predictions[0]

    def predict_from_file_and_save(self, image_path, destination_path):
        prediction = self.predict_from_file(image_path=image_path)
        save_image_from_tensor(image=prediction, file_path=destination_path, multiply=True)

