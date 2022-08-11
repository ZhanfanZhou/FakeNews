import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from src.build_model import Classifier
from src.utils import DataReader, analyze_data
from src.encode_feature import FeatureEncoder
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="svm")
    parser.add_argument("--data_path", type=str, default="../data/train.csv")
    parser.add_argument("--test_path", type=str, default="../data/test.csv")
    parser.add_argument("--test_label_path", type=str, default="../data/labels.csv")
    parser.add_argument("--log_path", type=str, default="../out")
    parser.add_argument("--seed", type=int, default=512)
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--log", action='store_true')
    parser.add_argument('--ngram', nargs='+', type=int, default=(1, 1))
    parser.add_argument('--svm_c', type=float, default=0.9)
    parser.add_argument('--svm_gamma', type=str, default='scale')
    parser.add_argument('--svm_kernel', type=str, default='poly')
    args = parser.parse_args()
    # load dataset
    print('loading data')
    X_train, X_test, Y_train, Y_test = DataReader.load_data(args)
    # analyze_data(X_train, Y_train)
    # featurize
    X_train, X_test = FeatureEncoder(args).encode(X_train, X_test)
    # training and testing
    print('training')
    classifier = Classifier(args, X_train, Y_train)
    pred = classifier.predict(X_test)
    # evaluate on validation set
    print('training done')
    classifier.evaluate(pred, Y_test)


if __name__ == '__main__':
    main()
