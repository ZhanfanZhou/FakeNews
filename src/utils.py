import datetime
import json
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_TOKEN = "[EMPTY]"


class DataReader:
    """a Helper class to load data from file"""
    def __init__(self):
        pass

    @staticmethod
    def load_data(args):
        """
        load training data or test data
        :return trainable dataframe objects
        """
        X_train, Y_train = DataReader._read_training(args.data_path)
        if args.test:
            X_test, Y_test = DataReader._read_test(args.test_path, args.test_label_path)
            return X_train, X_test, Y_train, Y_test

        X_train, X_vali, Y_train, Y_vali = train_test_split(X_train, Y_train, test_size=0.2, random_state=args.seed)
        return X_train, X_vali, Y_train, Y_vali

    @staticmethod
    def _read_training(train_src: str):
        """
        Read in the training file. The NaN values are dropped.
        :param train_src: path to the training file
        :return: DataFrame objects for samples and labels
        """
        articles, labels = DataReader._transform_training(train_src)
        articles, labels = pd.DataFrame(articles, dtype=str, columns=['article']), pd.DataFrame(labels, dtype=int)
        articles.dropna(inplace=True)
        return articles, labels

    @staticmethod
    def _read_test(test_src: str, label_src: str):
        """
        Read in test csv and ground true labels.
        :param test_src: path to test file
        :param label_src: path to labels
        :return: DataFrame objects for samples and labels
        """
        return DataReader._transform_test(test_src, label_src)

    @staticmethod
    def _transform_training(train_src: str):
        """
        pre-process and collate training samples from file.
        The ids are strictly increasing by 1, the end of a article is defined by the following rules:
        1, it contains a label at the end;
        2, the next line is a new sample or the end of the file.
        :param train_src: path to training csv file
        :return: a list of articles, a list of labels
        """
        articles, labels = [], []
        with open(train_src, 'r') as f:
            # skip the header
            next(f)
            line = f.readline()
            aid, text, label = None, None, None
            while line:
                # if a new article is found
                if aid is None:
                    if _contains_label(line):
                        # this is a one-liner sample;
                        # first split the line by the left most ","
                        _, rest = line.split(",", maxsplit=1)
                        # then split the rest by the right most ","
                        text, _, label = rest.strip().rpartition(",")
                        articles.append(text)
                        labels.append(label)
                    else:
                        aid, text = line.split(",", maxsplit=1)
                    line = f.readline()

                # a article is under processing
                else:
                    # if the current line could be the last line of a article
                    if _contains_label(line):
                        # if a new tweet is found in the next line, we are sure this is the end of a article
                        next_line = f.readline()
                        if not next_line or next_line.startswith(str(int(aid)+1)+','):
                            # split the line by the right most ","
                            more_text, _, label = line.rpartition(',')
                            text += more_text.strip()
                            line = next_line
                            articles.append(text)
                            labels.append(label)
                            aid = None
                            continue
                    # not the end of current article
                    text += line
                    line = f.readline()
        return articles, labels

    @staticmethod
    def _transform_test(test_src: str, label_src: str):
        """
        Load test data, labels from file.
        For the test data:
        1. Title, author, text columns are merged to form "article" column
        2. The NaN values are filled with DEFAULT_TOKEN.
        :param test_src: path to test file
        :param label_src: path to labels
        :return: an article dataframe and a label dataframe
        """
        labels = pd.read_csv(label_src, header=0, encoding='utf-8', dtype=int)
        articles = pd.read_csv(test_src, header=0, encoding='utf-8', dtype=str)
        articles['article'] = articles['title'] + articles['author'] + articles['text']
        articles.drop(['id', 'title', 'author', 'text'], axis=1, inplace=True)
        articles.fillna(DEFAULT_TOKEN, inplace=True)
        labels.drop(['id'], axis=1, inplace=True)
        return articles, labels


def _contains_label(text: str):
    """
    check if the given text contains label at the end, if yes it may be the end of a sample.
    :param text: a given line
    :return: a boolean value, if True, the given text contains label information
    """
    text = text.strip()
    return text.endswith(",1") or text.endswith(",0")


def save_log(args, result: str):
    """
    save args and result to args.log_path
    """
    time = datetime.datetime.now().strftime('%m-%d %H:%M:%S')
    log_name = f'{args.log_path}/{time}.log'
    with open(log_name, 'w') as f:
        f.write(result + '\n')
        f.write(json.dumps(vars(args))+'\n')

