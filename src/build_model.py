from src.utils import save_log
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics


class Classifier:
    def __init__(self, args, X, Y):
        self.args = args
        Y = Y.to_numpy().ravel()
        X = X.toarray()
        if 'svm' in args.model:
            self.model = SVC(C=0.8, gamma='scale', kernel='poly').fit(X, Y)
        if 'nb' in args.model:
            self.model = GaussianNB().fit(X, Y)

    def predict(self, X):
        return self.model.predict(X.toarray())

    def evaluate(self, Y_hat, Y):
        result = metrics.classification_report(Y, Y_hat, digits=3, output_dict=False)
        print(result)
        if self.args.log:
            save_log(self.args, result)

