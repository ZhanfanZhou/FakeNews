from sklearn.feature_extraction.text import CountVectorizer


class FeatureEncoder:
    def __init__(self, args):
        self.args = args

    def encode(self, X_train, X_test):
        vectorizer = CountVectorizer(analyzer='word', ngram_range=self.args.ngram, min_df=1, stop_words='english',
                                     max_features=2048)
        X_train = vectorizer.fit_transform(X_train['article'])
        X_test = vectorizer.transform(X_test['article'])
        return X_train, X_test

