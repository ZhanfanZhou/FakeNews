from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics


class Classifier:
    def __init__(self, args, X, Y):
        Y = Y.to_numpy().ravel()
        if 'svm' in args.model:
            self.model = SVC(C=0.5, gamma='scale', kernel='rbf').fit(X, Y)
        if 'nb' in args.model:
            X = X.toarray()
            self.model = GaussianNB().fit(X, Y)

    def predict(self, X):
        return self.model.predict(X.toarray())

    def evaluate(self, Y_hat, Y):
        print(metrics.classification_report(Y, Y_hat, digits=3, output_dict=False))

