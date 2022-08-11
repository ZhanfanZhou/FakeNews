from src.utils import save_log
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics


class Classifier:
    """
    A class for initializing and fitting given data to a svm or naive bayes classifier.
    This class also provides functions to predict samples and evaluate results
    """
    def __init__(self, args, X, Y):
        """
        :param args: training arguments for SVMs
        :param X: encoded numpy array feature vectors in shape (num_samples, num_features)
        :param Y: dataframe object with labels, in shape (num_samples, 1)
        """
        self.args = args
        Y = Y.to_numpy().ravel()
        if 'svm' in args.model:
            self.model = SVC(C=args.svm_c, gamma=args.svm_gamma, kernel=args.svm_kernel).fit(X, Y)
        if 'nb' in args.model:
            self.model = GaussianNB().fit(X, Y)

    def predict(self, X):
        """
        Perform prediction on samples in X
        :param X: encoded numpy array feature vectors
        :return: numpy array predictions in shape (num_samples, )
        """
        return self.model.predict(X)

    def evaluate(self, Y_hat, Y):
        """
        Show the classification result, save it to log if --log is specified
        :param Y_hat: numpy array predictions in shape (num_samples, )
        :param Y: numpy array predictions in shape (num_samples, ) or dataframe with labels, in shape (num_samples, 1)
        """
        result = metrics.classification_report(Y, Y_hat, digits=3, output_dict=False)
        print(result)
        if self.args.log:
            save_log(self.args, result)
