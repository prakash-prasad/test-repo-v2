import numpy as np
import keras
from keras import models
from src.config import PathConfig


# Read saved test data
def get_test_data(config):
    # x_test = np.load('/Users/prakash.prasad/Desktop/masters/sem3/mlops/test-repo-v2/data/test_independent.npy')
    # y_test = np.load('/Users/prakash.prasad/Desktop/masters/sem3/mlops/test-repo-v2/data/test_dependent.npy')
    x_test = np.load(config.test_independent_data_path)
    y_test = np.load(config.test_dependent_data_path)
    return x_test, y_test


config = PathConfig()
x_test, y_test = get_test_data(config)

# Read saved model
model = models.load_model(config.classifier_model_path)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=66)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')
