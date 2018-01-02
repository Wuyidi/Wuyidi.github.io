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
>>> from libsvm.svm import *
```

如果无报错说明安装成功

