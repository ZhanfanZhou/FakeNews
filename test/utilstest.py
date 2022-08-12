import unittest
from src.utils import DataReader, _contains_label
import pandas as pd
from argparse import Namespace


class UtilsTest(unittest.TestCase):
    """
    Test for utils.py
    """
    def test__read_training(self):
        """
        Test if the DataReader read the training file correctly
        """
        X, Y = DataReader._read_training('../data/train.csv')
        self.assertEqual(X.shape[0], Y.shape[0])
        self.assertEqual(X.shape[0], 20800)

    def test__read_test(self):
        """
        Test if the DataReader read the test file correctly
        """
        X, Y = DataReader._read_test('../data/test.csv', '../data/labels.csv')
        self.assertEqual(X.shape[0], 5200)

    def test__contain_label(self):
        text = "Text\n,1"
        self.assertTrue(_contains_label(text))
        text = "Text,0\n"
        self.assertTrue(_contains_label(text))

    def test__transform_test(self):
        X, Y = DataReader._transform_test('../data/test.csv', '../data/labels.csv')
        self.assertIsNotNone(X)
        self.assertIsNotNone(Y)

    def test__transform_training(self):
        X, Y = DataReader._read_training('../data/train.csv')
        self.assertIsNotNone(X)
        self.assertIsNotNone(Y)

    def test_load_data(self):
        args = Namespace()
        args.test = True
        args.data_path = '../data/toy_train.csv'
        args.test_path = '../data/test.csv'
        args.test_label_path = '../data/labels.csv'
        x1, x2, y1, y2 = DataReader.load_data(args)
        self.assertIsNotNone(args)
        self.assertIs(type(x1), pd.DataFrame)
        self.assertIs(type(x2), pd.DataFrame)
        self.assertIs(type(y1), pd.DataFrame)
        self.assertIs(type(y2), pd.DataFrame)
