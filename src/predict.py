import numpy as np
from keras import models
from src.config import PathConfig


class RandomMnistGenerator:
    def __init__(self, seed_):
        self.seed_ = seed_

    def generate_sparse_array(self, len):
        np.random.seed(self.seed_)
        shape = (len, 28, 28, 1)
        random_array = np.random.rand(*shape)
        threshold = 0.1
        # Generate a random array with values between 0 and 1. threshold for it to look like mnist
        sparse_array = np.where(random_array < threshold, np.random.uniform(0, 1), 0)
        print('Size of sparse array: {}'.format(sparse_array.shape))
        return sparse_array


if __name__ == '__main__':
    # Read saved model
    config = PathConfig()
    print('loading model from: {}'.format(config.classifier_model_path))
    model = models.load_model(config.classifier_model_path)

    # Generate mnist like sparse array
    prediction_data_generator = RandomMnistGenerator(seed_=3)
    prediction_array = prediction_data_generator.generate_sparse_array(len=20)

    # Evaluate the model
    prediction = model.predict(prediction_array)
    best_classes = np.argmax(prediction, axis=1)
    print(f'Prediction on random array: {best_classes}')
