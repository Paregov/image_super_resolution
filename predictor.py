from data_loader import load_image_file
from data_saver import save_image_from_tensor


class Predictor:
    def __init__(self, model, checkpoint_path):
        self._model = model
        self._model.load_weights(checkpoint_path)

    def predict_from_file(self, image_path):
        img = load_image_file(file_name=image_path, normalize=True)
        prediction = self._model.predict(img)
        return prediction

    def predict_from_file_and_save(self, image_path, destination_path):
        save_image_from_tensor(self.predict_from_file(image_path=image_path), destination_path, True)

