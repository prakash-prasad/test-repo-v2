import time
import numpy as np
import mlflow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
#  BatchNormalization
from src.config import PathConfig
from src.model_training import get_train_data
import os

if __name__ == '__main__':
    config = PathConfig()
    x_train, y_train = get_train_data(config)

    # Define the CNN model
    model1 = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    hyp_dict = {'optimizer': 'adam', 'loss_function': 'categorical_crossentropy',
                'batch_size': 128, 'epochs': 3, 'dropout_rate': 0.5}

    # Compile model
    model1.compile(optimizer=hyp_dict['optimizer'], loss=hyp_dict["loss_function"], metrics=['accuracy'])

    # Configure MLflow to log to a specific directory
    mlflow.set_tracking_uri(f"file:///{config.mlflow_tracking_path}")
    print("Tracking URI:", mlflow.get_tracking_uri())       # Print tracking URI

    # Set experiment name
    mlflow.set_experiment("mnist_cnn_experiment_v2")
    print("Experiment Name:", mlflow.get_experiment_by_name("mnist_cnn_experiment_v2"))

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("optimizer", hyp_dict['optimizer'])
        mlflow.log_param("loss_function", hyp_dict["loss_function"])
        mlflow.log_param("batch_size", hyp_dict['batch_size'])
        mlflow.log_param("epochs", hyp_dict['epochs'])
        mlflow.log_param("dropout_rate", hyp_dict['dropout_rate'])

        # Train the model
        start_time = time.time()
        print('epochs count {}'.format(hyp_dict['epochs']))
        history = model1.fit(x_train, y_train, batch_size=hyp_dict['batch_size'], epochs=hyp_dict['epochs'],
                            verbose=1, validation_split=0.1)
        training_time = time.time() - start_time

        # Log metrics
        train_loss, train_accuracy = model1.evaluate(x_train, y_train, verbose=0)
        mlflow.log_metric("training_loss", train_loss)
        mlflow.log_metric("training_accuracy", train_accuracy)
        mlflow.log_metric("training_time", training_time)

        # # Create an input example for the model
        # input_example = np.expand_dims(x_train[0], axis=0)

        # Log model
        # mlflow.keras.log_model(model1, "mnist_classifier_mlflow_new_v1", input_example=input_example)
        mlflow.keras.log_model(model1, "mnist_classifier_mlflow_new_v1")
        print(f"Model trained with training accuracy: {train_accuracy}, training loss: {train_loss}")
