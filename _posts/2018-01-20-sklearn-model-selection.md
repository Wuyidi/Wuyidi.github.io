---
layout:     post
title:      "Model selection"
subtitle:   "Choosing the right estimator"
date:       2018-01-20
author:     "Yidi"
header-img: "img/post-bg-2015.jpg"
tags:
    - machine learning
---



> *All models are wrong, but some models are useful. — George Box (Box and Draper 1987)*

### 介绍

本篇文章是系列文章Model selection and evaluation 的第一篇， 介绍了选择合适的机器学习算法



### 具体算法选择

根据[No free lunch theorem](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/No_free_lunch_theorem)，在机器学习中，不存在一个在各方面都最好的模型/算法，因为每一个模型都或多或少地对数据分布有先验的统计假设。取所有可能的数据分布的平均，每个模型的表现都一样好（或者一样糟糕）。因此，我们需要针对具体的问题，找到最好的机器学习算法。





![ml_map](/img/in-post/ml/ml_map.png)





