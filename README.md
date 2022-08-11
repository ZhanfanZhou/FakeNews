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
>Note: to train a naive bayes model, specify ```--model nb```

Test a model on test data:
```
python3 main.py --test --data_path ../data/train.csv --model svm --log
```
Arguments:
```--test```: run test on test data instead of validation data;
```--log```: output log; ```--data_path```: path to training data; ```--svm_c```, ```--svm_gamma```, ```--svm_kernel```: SVM parameters

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

Example:

*2,Why theTruth Might Get You Fired,Consortiumnews.com,"Why the Truth Might Get You Fired...*\
*By Lawrence Davidson...*\
*...*\
*Hope springs eternal.,1*\
*3,15 Civilians Killed In Single US Airstrike...*

Article 2 consists of 4 lines where the label (1) is shown in the last line. Article 3 starts at the fifth line.
The rules includes 1-4 lines as article 2.

2. Since the training dataset is balanced and all features are text, the features are concatenated to form a longer text.
3. The nan values are filled with a default token to prevent missing values.*
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
Given the training dataset is balanced, accuracy and F1 score could be used for evaluation. However, when dealing with real world data, positive samples are rarer.
F1 score is preferable to accuracy.

The cost expense of type I error and type II error are unclear, so the precision, recall and F1 score are listed.

Belows are F1 scores on test data.

| model | hyper-param | Precision-0 | Recall-0 | Precision-1 | Recall-1 | F1 score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Naive bayes | - | 0.755 | 0.648 | 0.742 | 0.828 | 0.740 |
| SVM(poly) | c=0.98,gamma=scale | 0.923 | 0.583 | 0.738 | 0.960 | 0.775 |
| SVM(rbf) | c=0.98,gamma=scale | 0.651 | 0.626 | 0.690 | 0.679 | 0.653 |

Naive bayes performs evenly on precision of 2 classes, whereas recall of 0 class drags down the performance. Overall, Naive bayes shows fair performance.

Poly-nominal SVM achieves outstanding result in precision for 0 class and recall for 1 class. However, the recall and precision for the counterpart drop severely due to the trade off between precision and recall.

RBF SVM requires further parameters tuning since theoretically it can simulate the poly kernel performance.
## what's next

### parameter tuning
It is possible to perform a grid search to find the optimal hyper-params.

It would be helpful to put training arguments to a yaml file for parameter tuning.
### handle overfitting
The preliminary experiments on validation data show both SVMs and Naive bayes overfit the training data easily. The performances drop significantly when models are tested on test data compared with validation data.

Increasinthe data variety, performing feature selection, adding regularization could help in our scenario.

### BERT fine-tuning
BERT bases on the Transformer which takes good care of long input sequence whereas RNN like models may suffer from long dependency issue.
### other possible improvement
* making use of the meta info such as author: if possible, learn author embeddings. The hypothesis is some authors are fake news maker.

* The text length is limited due to limited computation resources. For a long text classification task, one may use a sliding window on a long doc (with overlaps).
The sliding window divides the doc into a few parts. Each part, treated as a single doc, is fed into a model. The decision is made by aggregating all sub-docs.
