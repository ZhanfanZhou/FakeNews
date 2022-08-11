import unittest
from src.utils import DataReader


class UtilsTest(unittest.TestCase):
    """
    Test for utils.py
    """
    def test_read_training(self):
        """
        Test if the DataReader read the training file correctly
        """
        X, Y = DataReader._read_training('../data/train.csv')
        self.assertEqual(X.shape[0], Y.shape[0])
        self.assertEqual(X.shape[0], 20800)

    def test_read_test(self):
        """
        Test if the DataReader read the test file correctly
        """
        X, Y = DataReader._read_test('../data/test.csv', '../data/labels.csv')
        self.assertEqual(X.shape[0], 5200)
