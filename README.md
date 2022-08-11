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
python3 main.py --data_path ../data/train.csv --model svm
```
>Note: to train a naive bayes model, specify ```--model nb```. To know more about the parameters, see main.py.

Test a model on test data:
```
python3 main.py --test --data_path ../data/train.csv --model svm --log
```

## Dataset overview

The training dataset has over 20,000 articles with fields for the article title, author, and text.
The label field is the outcome variable where a 1 indicates the article is unreliable and a 0 indicates the article is reliable.
The ratio between positive and negative class is balanced.
The average article length is around 780.

| dataset | 0 class | 1 class | total |
| :---: | :---: | :---: | :---: |
| train.csv | 10413 | 10387 | 20800 |
| test.csv | - | - | 5200 |


## Data preprocessing
1. An article is a long document that consists of title, author, and text where text may contain multiple lines of sentences. To determine
the boundary of an article, rules below are defined to detect article boundary:
    * the last line of an article contains a label at the end;
    * the next line is an new article with a "strictly increasing by 1" #id or the end of the file.

2. Since the training dataset is balanced and all features are text, the features are concatenated to form a longer text.
3. The nan values are filled with a default token to prevent missing values.
<!--- 
The max article length is limited to 128 where the exceeding part is truncated and the missing part is filled with padding.

Note: to accelerate BERT tokenizer, the text is pre-trimmed.
-->

## The models
Two models: a naive bayes classifier and a SVM classifier are trained.

* Naive bayes is often used for spam detection, similar to Fake news detection.
* SVM is a strong baseline for classification problem, it is faster than neural net in general.

The n-gram features are extracted, which results in over 3000 dimensions feature vectors. The long articles alleviate the matrix sparsity issue to a certain extent.
To reduce the computation overhead and avoid over-fitting the data, the feature dimension is trimmed.



<!--
The pre-trained BERT base model is fine-tuned; the representation of [CLS] token in the final layer is used for classification.\
optimizer: Adam\
loss function: Cross entropy loss
-->


## Evaluation
Since the training dataset is balanced, and type I error and type II error are equally important, accuracy and F1 score could be used for evaluation. However, when dealing with real world data, positive samples are rarer.
F1 score is preferable to accuracy.

Belows are F1 scores of validation data and test data respectively.

| model | hyper-param | F1 score |
| :---: | :---: | :---: |
| Naive bayes | - | 0.727 |
| SVM(poly) | c=0.9,gamma=scale | 0.773 |
| SVM(rbf) | c=0.9,gamma=scale | 0.650 |
## what's next

### parameter tuning
It is possible to perform a grid search to find the optimal hyper-params.
### handle overfitting
The preliminary experiments on validation data show both SVMs and Naive bayes overfit the training data easily. The performances drop significantly when models are tested on test data compared with validation data.

Increasinthe data variety, performing feature selection, adding regularization could help in our scenario.

### BERT fine-tuning
BERT bases on the Transformer which takes good care of long input sequence whereas RNN like models may suffer from long dependency issue.
### other possible improvement
* making use of the meta info such as author: if possible, learn author embeddings. The hypothesis is some authors are fake news maker.

* The text length is limited due to limited computation resources. For a long text classification task, one may use a sliding window on a long doc (with overlaps).
The sliding window divides the doc into a few parts. Each part, treated as a single doc, is fed into a model. The decision is made by aggregating all sub-docs.


