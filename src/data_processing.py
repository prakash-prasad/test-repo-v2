import numpy as np

from keras.datasets import mnist
from keras.utils import to_categorical

from src.config import PathConfig

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape and normalize the data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print('shapes of x train and test, y train and test v1: ')
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
train_data_size, test_data_size = 500, 100
x_train = x_train[:train_data_size]
y_train = y_train[:train_data_size]
x_test = x_test[:test_data_size]
y_test = y_test[:test_data_size]
print('shapes of x train and test, y train and test v2: ')
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# np.save('/Users/prakash.prasad/Desktop/masters/sem3/mlops/test-repo-v2/data/train_independent.npy', x_train)
# np.save('/Users/prakash.prasad/Desktop/masters/sem3/mlops/test-repo-v2/data/test_independent.npy', x_test)
# np.save('/Users/prakash.prasad/Desktop/masters/sem3/mlops/test-repo-v2/data/train_dependent.npy', y_train)
# np.save('/Users/prakash.prasad/Desktop/masters/sem3/mlops/test-repo-v2/data/test_dependent.npy', y_test)

config = PathConfig()
np.save(config.train_independent_data_path, x_train)
np.save(config.train_dependent_data_path, y_train)
np.save(config.test_independent_data_path, x_test)
np.save(config.test_dependent_data_path, y_test)
