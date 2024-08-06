import time
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from src.config import PathConfig


def get_train_data(config):
    # x_train = np.load('/Users/prakash.prasad/Desktop/masters/sem3/mlops/test-repo-v2/data/train_independent.npy')
    # y_train = np.load('/Users/prakash.prasad/Desktop/masters/sem3/mlops/test-repo-v2/data/train_dependent.npy')
    x_train = np.load(config.train_independent_data_path)
    y_train = np.load(config.train_dependent_data_path)
    return x_train, y_train


config = PathConfig()
x_train, y_train = get_train_data(config)


# Define the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

init = time.time()
model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.1)
print('Time taken for training: {}'.format(time.time() - init))

# Saving model
model_save_path = config.classifier_model_path
print('Saving model on: {}'.format(model_save_path))
model.save(model_save_path)

# Check train loss
score = model.evaluate(x_train, y_train, verbose=0)
print(f'Train loss: {score[0]}')
print(f'Train accuracy: {score[1]}')