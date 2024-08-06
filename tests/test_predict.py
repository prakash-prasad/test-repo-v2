from unittest import TestCase
from unittest.mock import patch

import pytest
from src.predict import RandomMnistGenerator


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