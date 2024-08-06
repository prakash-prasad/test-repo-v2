import pytest
import numpy as np
# from src.predict import RandomMnistGenerator


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


@pytest.fixture(scope="session", autouse=True)
def my_fixture():
    print("Setting up fixture (once per session)")
    yield
    print("Tearing down fixture (once per session)")


@pytest.mark.usefixtures("my_fixture")
class TestRandomMnistGenerator:
    def test_my_test1(self):
        obj = RandomMnistGenerator(seed_=551)
        obj_array = obj.generate_sparse_array(len=50)
        assert obj_array.shape == (50, 28, 28, 1)
        print("Running test 1")

    def test_my_test2(self):
        obj = RandomMnistGenerator(seed_=5100)
        obj_array = obj.generate_sparse_array(len=7)
        assert obj_array.shape == (7, 28, 28, 1)
        print("Running test 2")
