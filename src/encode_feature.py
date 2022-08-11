from sklearn.feature_extraction.text import CountVectorizer


class FeatureEncoder:
    """
    A class for feature encoding, provides functions to extract features from samples
    """
    def __init__(self, args):
        """
        :param args: ngram argument for CountVectorizer
        """
        self.args = args

    def encode(self, X_train, X_test):
        """
        Encode articles in X_train, X_test to word count vectors
        :param X_train: dataframe
        :param X_test: same as X_train
        :return: numpy array feature vectors for training and testing in shape (num_samples, num_features)
        """
        vectorizer = CountVectorizer(analyzer='word', ngram_range=self.args.ngram, min_df=1, stop_words='english',
                                     max_features=1029)
        # get features
        X_train = vectorizer.fit_transform(X_train['article'])
        X_test = vectorizer.transform(X_test['article'])
        # convert the sparse matrix to dense
        X_train = X_train.toarray()
        X_test = X_test.toarray()
        return X_train, X_test

