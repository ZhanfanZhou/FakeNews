# Fake news!
***
Detect Fake news.
## requirements.txt
```
scikit-learn==0.21.2
numpy==1.19.5
pandas==1.1.0
```

## Structure
* /data: dataset
  * /train.csv labeled data
  * /test.csv test data.
  * /labels.csv ground true labels for testing.
  * /toy_train_csv: a toy training data set
* /test: unit tests
* /out: store outputted models
* /src: 
  * /main.py: run the model 
  * /build_model.py: define classifiers
  * /encode_feature.py: featurization
  * /utils.py: utils for loading data and dataset analysis
## Run it
train a svm classifier, run it on validation data:
```
python3 main.py --data_path ./data/train.csv --model svm
```
>Note: to train a naive bayes model, specify ```--model nb```. To know more about the parameters, see main.py.

Test a model on test data:
```
python3 main.py --test --data_path ./data/train.csv --model svm
```

## Dataset overview

The training dataset has over 20,000 articles with fields for the article title, author, and text. The label field is the outcome variable where a 1 indicates the article is unreliable and a 0 indicates the article is reliable.
* train.csv contains 20800 articles. The ratio between positive and negative is balanced.
Some fields have missing values. The average article length is around 700.
| class | count |
| :---: | :---: |
| 0 | 10387 |
| 1 | 10413 |

| dataset | count |
| :---: | :---: |
| train.csv | 10387 |
| test.csv | 10413 |



## Data preprocessing
An article is a long document consists of title, author, and text where text may contain multiple lines of sentences. To determine
the boundary of a article, rules belows are defined to detect article boundary:
* the last line of an article contains a label at the end;
* the next line is an new article with a "strictly increasing by 1" #id or the end of file.

Since the training dataset is balanced, and feature fields are text, the whole feature fields are concatenated to form a longer text.
The max article length is limited to 128 where the exceeding part is truncated and the missing part is filled with padding.

Note: to accelerate BERT tokenizer, the text is pre-trimmed.

## System description


### the model
The pre-trained BERT base model is fine-tuned; the representation of [CLS] token in the final layer is used for classification.\
optimizer: Adam\
loss function: Cross entropy loss



## Evaluation
Since the training dataset is balanced, and type I error and type II error are equally important, accuracy and F1 score could be used for evaluation. However, when dealing with real world data, positive samples are rarer. F1 score is more preferable than accuracy.\
Belows are F1 scores of validation data and test data respectively.
| model | hyper-param | F1 score |
| :---: | :---: | :---: |
| 1 | batch_size=32,lr=1e-5,epoch=1 | 0.998 |
| 2 | batch_size=32,lr=1e-5 | 66.6 |
| Naive bayes | - | 0.802 |
| SVM | c=0.5,\gamma= | 0.95 |
## QA

### why BERT
BERT bases on Transformer which takes good care of long input sequence whereas RNN like models may suffer from long dependency issue.
### parameter tuning
BERT fine-tuning requires huge computation overhead, a validation dataset is split from the training dataset for hyper-parameter tuning.
### possible improvement
Other than the parameter tuning, learning tricks such as scheduler etc., we may consider:
* making use of the meta info such as author: if possible, learn author embeddings. The hypothesis is some authors are fake news maker.

* The text length is limited due to limited computation resources. For a long text classification task, one may use a sliding window on a long doc (with overlaps).
The sliding window divides the doc into a few parts. Each part, treated as a single doc, is fed into a model. The decision is made by aggregating all sub-docs.


