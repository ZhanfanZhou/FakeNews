# Fake news!
***
Detect Fake news.
## requirements.txt & Environment
```
scikit-learn==0.21.2
numpy==1.19.5
pandas==1.1.0
```
python3.6

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
To train a svm classifier and test it on validation data, run:
```
python3 main.py --data_path ../data/train.csv --model svm
```
>To specify hyper-parameters for SVM, use
```--svm_c```, ```--svm_gamma```, ```--svm_kernel```

To train a Naive bayes classifier and test it on validation data, run:
```
python3 main.py --data_path ../data/train.csv --model nb
```

To test a model on test data:
```
python3 main.py --test --data_path ../data/train.csv --model svm --log
```
Arguments:

```--model ```: specify nb or svm

```--test```: run test on test data instead of validation data;

```--log```: output log; 

```--data_path```: path to training data;

To know more about the details, refer to main.py

## Dataset overview

The training dataset has over 20,000 articles with fields for the article title, author, and text.

The label field is the outcome variable where a 1 indicates the article is unreliable and a 0 indicates the article is reliable.
The ratio between positive and negative class is balanced.


| dataset | 0 class | 1 class | total |
| :---: | :---: | :---: | :---: |
| train.csv | 10413 | 10387 | 20800 |
| test.csv | 2339 | 2861 | 5200 |

The length of articles are inspected:

![alt text](https://github.com/ZhanfanZhou/FakeNews/blob/master/data.png)

The average article length is around 780.

## Data preprocessing
Step: 1. An article is a long document that consists of title, author, and text where text may contain multiple lines of sentences. To determine
the boundary of an article, the rules below are defined to detect article boundary:

  * the last line of an article contains a label at the end;
  * the next line is a new article with a "strictly increasing by 1" #id or the end of the file.

Example:

Before step 1:

| id | title | author | text | label |
| :---: | :---: | :---: | :---: | :---: |
| 1 | "FLYNN: Hillary Clinton... | Daniel J. Flynn | "Ever get the feeling your... | 0 |
| 2 | Why the Truth Might... | Consortiumnews.com | "Why the Truth Might Get... |  |
|   |  |  | The tension between... |  |
|   |  |  | ...Hope springs eternal." | 1 |
| 3 | 15 Civilians Killed... | Jessica Purkiss | "Videos 15 Civilians... |  |
After step 1:

| id | title | author | text | label |
| :---: | :---: | :---: | :---: | :---: |
| 1 | "FLYNN: Hillary Clinton... | Daniel J. Flynn | "Ever get the feeling your... | 0 |
| 2 | Why the Truth Might... | Consortiumnews.com | "Why the Truth Might Get... | 1 |
| 3 | 15 Civilians Killed... | Jessica Purkiss | "Videos 15 Civilians... | 1 |

Step 2. Since the training dataset is balanced and all features are text, the features are concatenated to form a longer text.

Example after step 2:

| id | article | label |
| :---: | :---: | :---: |
| 1 | "FLYNN: Hillary Clinton... | 0 |
| 2 | Why the Truth Might... | 1 |
| 3 | 15 Civilians Killed... | 1 |


Step 3. The NaN values are filled with a default token to prevent missing values.

## The models
Base models: a naive bayes classifier 
* Naive bayes is often used for spam detection, similar to Fake news detection.

First iteration: a SVM classifier

* SVM is a strong baseline for classification problems, it is faster than neural nets in general.

### feature extraction
The n-gram features are extracted, which results in over 3000 dimensions feature vectors. The long articles alleviate the matrix sparsity issue to a certain extent.
To reduce the computation overhead and avoid over-fitting the data, the feature dimension is trimmed.


## Evaluation
Given the training dataset is balanced, accuracy and F1 score could be used for evaluation. However, when dealing with real world data, positive samples are rarer.
F1 score is preferable to accuracy.

The cost expense of type I error and type II error are unclear, so the precision, recall and F1 score are listed.

Below are Precision, recall and F1 scores on test data.

| model | hyper-param | Precision | Recall | F1 score |
| :---: | :---: | :---: | :---: | :---: |
| Naive bayes | - | 0.742 | 0.828 | 0.783 |
| SVM(poly) | c=0.98,gamma=scale | 0.738 | 0.960 | 0.835 |
| SVM(rbf) | c=0.98,gamma=scale | 0.690 | 0.679 | 0.685 |

Naive bayes performs evenly on precision and recall, with recall slightly better. Overall, Naive bayes shows fair performance.

Poly-nominal SVM obtains outstanding recall. However, the precision drops severely due to the trade-off between precision and recall.

RBF SVM requires further parameters tuning since theoretically it can simulate the poly kernel performance.
## what's next

### Model optimization
#### handling overfitting
The preliminary experiments on validation data(20% of the training data) show both SVMs and Naive bayes overfit the training data easily.

The performances(F1 score) drop significantly from 98.9% on validation data to around 70% on test data.

Increasing the data variety, performing feature selection, adding regularization could help in our scenario.
#### hyper-parameter tuning
It is possible to perform a grid search to find the optimal hyper-params.

### Feature expansion
Make use of the Meta info such as author. The hypothesis is some authors are fake news maker. If possible, learn author embeddings.

### Model exploration
Build a neural net and leverage transfer learning to help classification, such as BERT.
BERT is a powerful pre-trained language model that can be fine-tuned for down stream tasks.
