# BERT_Multi_Label_Classification
## BERT
## BERT.\_\_init__
```python
bert = (self, per_train_model_dir, num_of_classes, class_name_list=None, **kwargs)
```

> 初始化模型

* `per_train_model_dir`:原始预训练模型存储路径(**文件夹**)
* `num_of_classes`:分类数量
* `class_name_list`:`default=None`, 各分类对应名称，如果输入为`None`, 则默认分类名称为`[label_0, label_1, ...]`
* `**kwargs`:
    * `split_rate`:`default=0.8`, 训练集与验证集划分比例
    * `max_length`:`default=256`, 句子最大长度
    * `learning_rate`:`default=2e-5`, 学习率
    * `batch_size`:`default=64`, 每个batch的数量
    * `model_save_dir`:`default='ModelWeights'`, 模型权重保存路径
    * `line_acc`:`default=0.8`, 预测结果置信准确度

### bert.train

```python
bert.train(self, train_data_path, epochs, save_weight=True, checkpoint_path=None)
```

> 对模型进行训练

* `train_data_path`:训练数据存储路径(自定按照指定比例划分训练集与验证集)
* `epoch`:训练次数
* `save_weight`:`default=True`, 保存训练结果
* `checkpoint_path`:`default=None`, 断点保存路径, 如果不是`None`将从`checkpoint_path`中加载模型权重并开始训练

### BERT.predict

```python
bert.predict(self, test_data_path, output_path, model_path=None)
```

> 使用训练好的模型进行分类标记

* `test_data_path`:待标记数据文件存储路径
* `output_path`:标记结果文件保存路径
* `model_path`:`default=None`, 如果不是`None`将从`model_path`中重新加载模型权重
