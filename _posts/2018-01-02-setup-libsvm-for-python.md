---
layout:     post
title:      "Mac上如何安装Libsvm的python接口"
subtitle:   
date:       2018-01-02
author:     "Yidi"
header-img: "img/post-bg-2015.jpg"
tags:
    - machine learning 
    - svm
---



### 下载Libsvm

Libsvm 3.22 版本下载地址: [zip](http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+zip) ; [tar.gz](http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+tar.gz)



### 安装

```bash
cd libsvm-3.22
make
```

```bash
cd python
make
```

在terminal中打开python

```python
>>> import sys
>>> sys.path
```

将libsvm.so.2 移动到指定目录

```bash
mv libsvm.so.2 path
```

创建libsvm目录将python文件夹下的文件复制到该目录下

```bash
cd path
makdir libsvm
cd libsvm
mv libsvm-3.22/python/svm.py .
mv libsvm-3.22/python/svmutil.py .
cat > __init__.py
```



### 测试

```python
from libsvm.svmutil import *
from libsvm.svm import *


y, x = [1, -1], [{1: 1, 2: 1}, {1: -1, 2: -1}]
prob = svm_problem(y, x)
param = svm_parameter('-t 0 -c 4 -b 1')
model = svm_train(prob, param)


yt = [1]
xt = [{1: 1, 2: 1}]
p_label, p_acc, p_val = svm_predict(yt, xt, model)
print(p_label)
```

输出

> *
> optimization finished, #iter = 1
> nu = 0.062500
> obj = -0.250000, rho = 0.000000
> nSV = 2, nBSV = 0
> Total nSV = 2
> *
> optimization finished, #iter = 1
> nu = 0.062500
> obj = -0.250000, rho = 0.000000
> nSV = 2, nBSV = 0
> Total nSV = 2
> *
> optimization finished, #iter = 1
> nu = 0.062500
> obj = -0.250000, rho = 0.000000
> nSV = 2, nBSV = 0
> Total nSV = 2
> *
> optimization finished, #iter = 1
> nu = 0.062500
> obj = -0.250000, rho = 0.000000
> nSV = 2, nBSV = 0
> Total nSV = 2
> Model supports probability estimates, but disabled in predicton.
> Accuracy = 100% (1/1) (classification)
> [1.0]