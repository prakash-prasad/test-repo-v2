import os


class PathConfig:
    def __init__(self):
        self.prev_dir = os.path.dirname(os.getcwd())
        self.train_independent_data_path = os.path.join(self.prev_dir, 'data/train_independent.npy')
        self.test_independent_data_path = os.path.join(self.prev_dir, 'data/test_independent.npy')
        self.train_dependent_data_path = os.path.join(self.prev_dir, 'data/train_dependent.npy')
        self.test_dependent_data_path = os.path.join(self.prev_dir, 'data/test_dependent.npy')

        self.classifier_model_path = os.path.join(self.prev_dir, 'models/mnist_classifier_v1.keras')
        self.mlflow_tracking_path = os.path.join(self.prev_dir, 'src/mlflow_tracking')
