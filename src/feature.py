from sklearn.feature_extraction.text import CountVectorizer


def get_feature(X_train, X_test):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), min_df=1, stop_words='english', max_features=2048)
    X_train = vectorizer.fit_transform(X_train['article'])
    X_test.info()
    X_test = vectorizer.transform(X_test['article'])
    return X_train, X_test

