---
layout:     post
title:      "Model evaluation"
subtitle:   "quantifying the quality of predictions"
date:       2018-01-21
author:     "Yidi"
header-img: "img/post-bg-2015.jpg"
tags:
    - machine learning 
    - sklearn
---



> 本篇文章是系列文章Model selection and evaluation 中的一篇，其中的例子来源于 [scikit-learn](http://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation) 的官方文档



### 介绍

在学习机器学习的过程中, 常常要对预测的结果进行合理的评价，本篇博客是用来记录一些常用的模型验证方法和结果评价。

sklearn 提供了三种不同的API 用来评估模型预测的质量：

- **Estimator score method**： 每种estimator都有一个`score` 的方法，为他们设计的问题提供一个默认的评估标准。
- **Score parameter**： 模型评估工具用了交叉验证（[cross-validation](http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)）依赖于内部 `scoring` 策略
- **Metric functions**：`metrics` 模块实现了一些函数，用来评估预测误差




### `scoring` 参数： 定义模型评估规则

模型选择和评估用的一些工具[`model_selection.GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) 和 [`model_selection.cross_val_score`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score)

#### 常用评分参数使用场景

##### 分类

1. accuracy : [`metrics.accuracy_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score)
2. average_precision : [`metrics.average_precision_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score)
3. f1 : [`metrics.f1_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)
4. f1_micro : [`metrics.f1_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)
5. f1_macro : [`metrics.f1_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)
6. f1_weighted : [`metrics.f1_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)
7. f1_samples : [`metrics.f1_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)
8. neg_log_loss : [`metrics.f1_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)
9. precision : [`metrics.precision_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score)
10. recall : [`metrics.recall_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score)
11. roc_auc : [`metrics.roc_auc_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score)

##### 聚类

1. adjusted_mutual_info_score : [`metrics.adjusted_mutual_info_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html#sklearn.metrics.adjusted_mutual_info_score)
2. adjusted_rand_score : [`metrics.adjusted_rand_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html#sklearn.metrics.adjusted_rand_score)
3. completeness_score : [`metrics.completeness_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html#sklearn.metrics.completeness_score)
4. fowlkes_mallows_score : [`metrics.fowlkes_mallows_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fowlkes_mallows_score.html#sklearn.metrics.fowlkes_mallows_score)
5. homogeneity_score : [`metrics.homogeneity_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html#sklearn.metrics.homogeneity_score)
6. mutual_info_score : [`metrics.mutual_info_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html#sklearn.metrics.mutual_info_score)
7. normalized_mutual_info_score : [`metrics.normalized_mutual_info_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html#sklearn.metrics.normalized_mutual_info_score)
8. v_measure_score : [`metrics.v_measure_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html#sklearn.metrics.v_measure_score)

##### 回归

1. explained_variance : [`metrics.explained_variance_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score)
2. neg_mean_absolute_error : [`metrics.mean_absolute_error`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error)
3. neg_mean_squared_error : [`metrics.mean_squared_error`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error)
4. neg_mean_squared_log_error : [`metrics.mean_squared_log_error`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html#sklearn.metrics.mean_squared_log_error)
5. neg_median_absolute_error : [`metrics.median_absolute_error`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error)
6. r2 : [`metrics.r2_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score)

##### 实例

```python
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf = svm.SVC(probability=True, random_state=0)
cross_val_score(clf, X, y, scoring='neg_log_loss')
model = svm.SVC()
cross_val_score(model, X, y, scoring='wrong_choice')
```



#### 使用metrics方法定义自己的评分策略

##### 实例1

```python
from sklearn.metrics import fbeta_score, make_scorer
ftwo_scorer = make_scorer(fbeta_score, beta=2)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
grid = GridSearchCV(LinearSVC(), param_grid={'C': [1,10]}, scoring=ftwo_scorer)
```



##### 实例2

```python
import numpy as np
from sklearn.dummy import DummyClassifier
def my_custom_loss_func(ground_truth, prdedcions):
    diff = np.abs(ground_truth - predictions).max()
    return np.log(1 + diff)

# loss_func will negate the return value of my_custom_loss_func,
# which will be np.log(2), 0.693, given the values for ground_truth
# and predictions defined below
loss = make_scorer(my_custom_loss_func, greater_is_better=False)
score = make_scorer(my_custom_loss_func, greater_is_better=True)
ground_truth = [[1], [1]]
predictions = [0, 1]
clf = DummyClassifier(strategy='most_frequent', random_state=0)
clf = clf.fit(ground_truth, predictions)
loss(clf,ground_truth, predictions) 
score(clf,ground_truth, predictions)
```

#### 多个metric评估

##### 定义参数

```python
scoring = ['accuracy', 'precision']
```

```python
from sklearn.metrics import accuarcy_score
from sklearn.metrics import make_scorer
scoring = {'accuracy': make_scorer(accuracy_score),
           'prec': 'precision'}
```

##### 实例

```python
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
# A sample toy binary classification dataset
X, y = datasets.make_classification(n_classes=2, random_state=0)
svm = LinearSVC(random_state=0)
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
scoring = {'tp' : make_scorer(tp), 'tn' : make_scorer(tn),
           'fp' : make_scorer(fp), 'fn' : make_scorer(fn)}
cv_results = cross_validate(svm.fit(X, y), X, y, scoring=scoring)
# Getting the test set true positive scores
print(cv_results['test_tp'])          

# Getting the test set false negative scores
print(cv_results['test_fn'])    
```



### 分类Metrics

 [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)定义并实现了一些工具模块和方法来评估分类器的表现



#### 准确率（Accuracy score）

 [`accuracy_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score)方法用来计算准确率，默认是百分制的，如果(normalize=False)则计算实际预测正确的数量。

预测值与真实值的准确率计算公式：

$$
accuracy(y,\hat{y} )=\frac{1}n_{samples}∑_{i=0}^{n_{samples}-1}1(\hat{y}_i=y_i)
$$

$$1(x)$$是[指示函数](https://en.wikipedia.org/wiki/Indicator_function)

```python
>>> import numpy as np
>>> from sklearn.metrics import accuracy_score
>>> y_pred = [0, 2, 1, 3]
>>> y_true = [0, 1, 2, 3]
>>> accuracy_score(y_true, y_pred)
0.5
>>> accuracy_score(y_true, y_pred, normalize=False)
2
```

如果是多标签的情况：

```python
>>> accuracy_score(np.array([[0,1],[1,1]]), np.ones((2, 2)))
0.5
```




#### 混淆矩阵（Confusion matrix）

输出：

```
Confusion matrix, without normalization
[[13 0 0]
 [0 10 6]
 [0  0 9]]
 
Normalized confusion matrix
[[1.    0.    0.  ]
 [0.    0.62  0.38]
 [0.    0.    1.  ]]
```



绘制混淆矩阵的函数

![confusion_matrix](/img/in-post/ml/confusion_matrix.png)

```python
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import itertools
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_truth, 
                          y_predict, 
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_truth, y_predict)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
```



#### 分类报告（Classification report）

[`classification_report`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report)方法构建了一个文本报告，用于展示主要的分类metrics。接下来的例子自定义了`target_names` 与推断的标签

```python
>>> from sklearn.metrics import classification_report
>>> y_true = [0, 1, 2, 2, 0]
>>> y_pred = [0, 0, 2, 1, 0]
>>> target_names = ['class0', 'class1', 'class2']
>>> print(classification_report(y_true, y_pred, target_names=target_names))
             precision    recall  f1-score   support

    class 0       0.67      1.00      0.80         2
    class 1       0.00      0.00      0.00         1
    class 2       1.00      1.00      1.00         2

avg / total       0.67      0.80      0.72         5

```



在下面的分类器优化例子中，我们使用了分类报告来展示了嵌套交叉验证来寻找最优参数的实验结果。

使用[`sklearn.model_selection.GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) 模块来寻找最优解

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


# Loading the Digits dataset
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
scores = ['percision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_macro' % score)
    clf.fit(X_train, y_train)
    
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f(+/-%0.03f) for %r") % (mean, std*2, params)
	print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred=y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
```



#### 查准率（Precision），查全率（Recall），F-measure

直观来讲，`precision ` 表现为分类器找的对，`recall` 表现为找的全， F-measure（$$F_1, F_\beta$$）



#### 二分类（Binary classification）

**precision** : $$\frac{tp}{tp + fp}$$

**recall**： $$\frac{tp}{tp + fn}$$

$$F_\beta$$:	$$(1 + \beta^2)\frac{precision \times recall}{\beta^2 precision + recall}$$

tp(true positive): 被判断为正样本，实际也是正样本。

fp(false positive): 判断为正样本，但事实上是负样本。

fn(False negative): 被判断为负样本，但事实上是正样本。

tn(True negative)：被判断为负样本，实际也是负样本。



#### 多分类与多标签（Multiclass and mutilabel classification）

在多标签和多分类的问题上，precision，recall，F-measure的概念可以独立运用到每一个标签上。

1. $$y$$ 是 （sample, label）的预测集

2. $$\hat{y}$$ 是（smaple, label）的真实集

3. $$L$$ 是标签集

4. $$S$$ 是样本集

5. $$y_s$$ 是$$y$$ 关于样本的子集 $$y_s:=\left\{(s^\acute{},l)\in{y\vert{s^\acute{}=s}}\right\}$$

6. $$y_l$$ 是$$y$$ 关于标签的子集

7. 同样 $$\hat{y_s}, \hat{y_l}$$ 是关于$$\hat{y}$$的子集

8. Precision： $$P(A, B):=\frac{\vert A\cap{B}\vert}{\vert A\vert}$$

9. Recall：$$R(A, B):=\frac{\vert A\cap{B}\vert}{\vert A\vert}$$ (当$$B = \emptyset$$ 则 $$R(A, B):= 0$$ 对于查准率$$P$$也是一样)

10. F-measure：$$F_{\beta}(A, B) := (1 + \beta ^ 2)\frac{P(A, B) \times R(A, B)}{\beta ^ 2P(A, B) + R(A,B)}$$

    ​

| average  | Precision                                                    | Recall                                                       | F_beta                                                       |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| micro    | $$P(y, \hat {y})$$                                           | $$R(y, \hat {y})$$                                           | $$F_\beta(y, \hat {y})$$                                     |
| samples  | $$\frac{1}{\vert S \vert} \sum_{s\in S}P(y_s, \hat{y_s})$$   | $$\frac{1}{\vert S \vert} \sum_{s\in S}R(y_s, \hat{y_s})$$   | $$\frac{1}{\vert S \vert} \sum_{s\in S}F_\beta(y_s, \hat{y_s})$$ |
| macro    | $$\frac{1}{\vert L \vert} \sum_{l\in L}P(y_l, \hat{y_l})$$   | $$\frac{1}{\vert L \vert} \sum_{l\in L}R(y_l, \hat{y_l})$$   | $$\frac{1}{\vert L \vert} \sum_{l\in L}F_\beta(y_l, \hat{y_l})$$ |
| weighted | $$\frac{1}{\sum_{t \in L}\vert \hat{y_l} \vert} \sum_{l\in L} \vert \hat {y_l}\vert P(y_l, \hat{y_l})$$ | $$\frac{1}{\sum_{t \in L}\vert \hat{y_l} \vert} \sum_{l\in L} \vert \hat {y_l}\vert R(y_l, \hat{y_l})$$ | $$\frac{1}{\sum_{t \in L}\vert \hat{y_l} \vert} \sum_{l\in L} \vert \hat {y_l}\vert F_\beta(y_l, \hat{y_l})$$ |
| None     | $$\left\langle P(y_l, \hat{y_l} \vert l \in L) \right \rangle$$ | $$\left\langle R(y_l, \hat{y_l} \vert l \in L) \right \rangle$$ | $$\left\langle F_\beta(y_l, \hat{y_l} \vert l \in L) \right \rangle$$ |

