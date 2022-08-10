from src.model import Classifier
from src.utils import DataReader
from src.feature import get_feature
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="svm")
    parser.add_argument("--data_path", type=str, default="../data/toy_train.csv")
    parser.add_argument("--test_path", type=str, default="../data/test.csv")
    parser.add_argument("--test_label_path", type=str, default="../data/labels.csv")
    parser.add_argument("--seed", type=int, default=1029)
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()

    # load dataset
    X_train, X_test, Y_train, Y_test = DataReader.load_data(args)
    # featurize
    X_train, X_test = get_feature(X_train, X_test)
    # training and testing
    classifier = Classifier(args, X_train, Y_train)
    pred = classifier.predict(X_test)
    # evaluate on validation set
    classifier.evaluate(pred, Y_test)


if __name__ == '__main__':
    main()
