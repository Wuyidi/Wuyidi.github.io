---
layout:     post
title:      "Class method vs Static method in Python"
subtitle:   "Python's Instance, Class, and Static Methods Demystified"
date:       2019-01-11
author:     "Yidi"
header-img: "img/Pythons-Instance-Class-and-Static-Methods-Demystified_Watermarked.webp"
tags:
    - python
---

> Python中的类也是一个普通对象，如果需要直接使用这个类，例如将类作为参数传递到其他函数中，又希望在实例化这个类之前就能提供某些功能，那么最简单的办法就是使用class method和static method。这两者的区别在于在存在类的继承的情况下对多态的支持不同。                                                                    —— [灵剑](https://www.zhihu.com/question/20021164/answer/537385841 )

### Overview

```python
class MyClass:
    def method(self):
        return 'instance method called', self

    @classmethod
    def classmethod(cls):
        return 'class method called', cls

    @staticmethod
    def staticmethod():
        return 'static method called'
```

### Instance Method

When creating an instance method, the first parameter is always `self`. You can name it anything you want, but the meaning will always be the same, and you should use `self`since it's the naming convention. `self` is (usually) passed hiddenly when calling an instance method; it represents the instance calling the method.

### Class Method

The @classmethod decorator, is a builtin function decorator that is an expression that gets evaluated after your function is defined. The result of that evaluation shadows your function definition.
A class method receives the class as implicit first argument, just like an instance method receives the instance.

The idea of class method is very similar to instance method, only difference being that instead of passing the instance hiddenly as a first parameter, we're now passing the class itself as a first parameter.

### Static Method

A static method does not receive an implicit first argument.

### When to use what?

- We generally use class method to create factory methods. Factory methods return class object ( similar to a constructor ) for different use cases.
- We generally use static methods to create utility functions.

下面的设计中我们试图通过插件的方式支持各类数据库作为底层数据储存，插件只需要实现DBCursor的子类即可。在`save_to_database`时，系统查找合适的插件类，通过这个类创建DBCursor，通过cursor将数据保存到数据库中。但是，一般数据库的插件都支持很多配置，我们希望这个配置可以以集中的方式保存在配置应用中，这样我们为DBCursor类增加了一个configure接口，它会在任何DB Cursor被实例化之前，首先在类上被调用，这样在初始化`__init__`的时候，就可以使用这个配置了。

```python
class DBCursor(Object):
    def __init__(self, **kwargs):
        pass
    def execute(self,sql):
        raise NotImplementedError
	@classmethod
    def configure(cls, config):
        cls.config = config

def bootstrap():
    # Load config from config file
    config = load_config()
    cursor_plugins = load_cursor_plugins()
    for p in cursor_plugins:
        if hasattr(p, 'configure'):
            # Get plugin related config
            plugin_cfg = get_plugin_config(config, p)
            # Send config to plugin
            p.configure(plugin_cfg)
        register_cursor_plugin(p)

def save_to_database(data):
    # Find suitable cursor plugin
    arguments = get_data_options(data)
    cursor_plugin = find_suitable_cursor(arguments, data)
    # Create new cursor
    cursor = cursor_plugin(**arguments)
    # Generate SQL
    sql = generate_sql(data, cursor)
    # Call execute
    cursor.execute(sql)
```

我们要将一个对象序列化成数据，再从数据中重新恢复时，一般来说，可以将序列化的方法作为instance method，而将恢复的方法作为class method，原因在于恢复时不一定经过普通的init，这样恢复过程可以看作是一种特殊的构造过程

```python
class JSONSerializable(object):
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v) #self.k = v
    
    def serialize(self):
        return json.dumps({k:v for k,v in self.__dict__.items() if not k.startswith('_')})
    
    @classmethod
    def deserialize(cls, data):
        obj = cls.__new__(cls)
        for k,v in json.loads(data).items():
            setattr(obj, k, v)
        return obj


def save_to_file(file, obj):
    write_class_information(file, type(obj))
    write_data(file, obj.serialize())

def load_from_file(file):
    info = read_class_information(file)
    cls = find_class(info)
    data = read_data(file)
    return cls.deserialize(data)
```

### Conclusion

Python中的classmethod（和staticmethod）并不止拥有美学上（或者命名空间上）的意义，而是可以实际参与多态的、足够纯粹的OOP功能，原理在于Python中类可以作为first class的对象使用，很大程度上替代其他OOP语言中的工厂模式。classmethod既可以作为factory method提供额外的构造实例的手段，也可以作为工厂类的接口，用来读取或者修改工厂类本身。classmethod还可以通过额外的类引用，提供继承时的多态特性，实现子类挂载点等。
