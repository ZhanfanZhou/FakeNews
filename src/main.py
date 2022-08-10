from src.model import Classifier
from src.utils import DataReader, analyze
from src.feature import FeatureEncoder
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="nb")
    parser.add_argument("--data_path", type=str, default="../data/train.csv")
    parser.add_argument("--test_path", type=str, default="../data/test.csv")
    parser.add_argument("--test_label_path", type=str, default="../data/labels.csv")
    parser.add_argument("--seed", type=int, default=1029)
    parser.add_argument("--test", action='store_true')
    parser.add_argument('--ngram', nargs='+', type=int, default=(1, 1))
    args = parser.parse_args()

    # load dataset
    X_train, X_test, Y_train, Y_test = DataReader.load_data(args)
    #
    analyze(X_train, Y_train)
    # featurize
    X_train, X_test = FeatureEncoder(args).encode(X_train, X_test)
    # training and testing
    classifier = Classifier(args, X_train, Y_train)
    pred = classifier.predict(X_test)
    # evaluate on validation set
    classifier.evaluate(pred, Y_test)


if __name__ == '__main__':
    main()
