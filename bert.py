#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/16 15:46
# @Author  : ZhouHang
# @Email   : zhouhang@idataway.com
# @File    : bert.py
# @Software: PyCharm
__all__ = ['BERT']

import os
import numpy
import pandas
import swifter
import datetime
import tensorflow as tf
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib.font_manager import FontProperties
from transformers.modeling_tf_utils import get_initializer
from transformers import TFBertPreTrainedModel, TFBertMainLayer, BertTokenizer


class TFBertForMultilabelClassification(TFBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super(TFBertForMultilabelClassification, self).__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.bert = TFBertMainLayer(config, name='bert')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(config.num_labels,
                                                kernel_initializer=get_initializer(config.initializer_range),
                                                name='classifier',
                                                activation='sigmoid')

    def call(self, inputs, **kwargs):
        outputs = self.bert(inputs, **kwargs)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=kwargs.get('training', False))
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        return outputs  # logits, (hidden_states), (attentions)


class BERT(object):
    def __init__(self, per_train_model_dir, num_of_classes, class_name_list=None, split_rate=0.8, max_length=128,
                 batch_size=32, learning_rate=2e-5, model_save_dir='ModelWeights', line_acc=0.8):
        '''

        :param per_train_model_dir: 预训练模型保存文件夹, str
        :param num_of_classes: 分类数量, int
        :param class_name_list: 各分类对应名称, list, default=None
        :param split_rate: 训练集验证集划分比例, float, default=0.8
        :param max_length: 句子最大长度, int, default=128
        :param batch_size: 学习率, float, default=2e-5
        :param learning_rate: 每个batch的数量, int, default=32
        :param model_save_dir: 模型权重保存路径, str, default='ModelWeights'
        :param line_acc: 预测结果置信准确度, float, default=0.8
        '''
        self._per_train_model_dir = per_train_model_dir
        self._num_of_classes = num_of_classes
        self._split_rate = split_rate
        self._max_length = max_length
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._model_save_dir = model_save_dir
        self._line_acc = line_acc
        self._tokenizer = BertTokenizer.from_pretrained(self._per_train_model_dir)
        if class_name_list is None:
            self._class_name_list = [f'label_{i}' for i in range(self._num_of_classes)]
        else:
            if isinstance(class_name_list, list):
                if len(class_name_list) == self._num_of_classes:
                    self._class_name_list = class_name_list
                else:
                    raise IndexError
            else:
                raise ValueError

        self.model = self.__init_model()

    def __str__(self):
        info = '<BERT.model.Multi-Label_Classification>'
        return info

    def __read_data(self, data_path):
        '''
        读取文件，支持*.csv、*.xlsx、*.xls文件类型

        :param data_path:
        :return:
        '''
        _, data_type = os.path.splitext(data_path)
        if data_type == '.csv':
            data = pandas.read_csv(data_path)
        elif data_type == '.tsv':
            data = pandas.read_csv(data_path, sep='\t')
        elif data_type == '.xlsx' or data_path == '.xls':
            data = pandas.read_excel(data_path)
        else:
            raise NameError('不支持读取此类型文件')
        return data

    def __convert_example_to_feature(self, review):
        '''
        结合步骤进行标记化、词条向量映射、添加特殊标记以及截断超过最大长度的评论

        :param review:
        :return: encode data
        '''

        return pandas.Series(
            self._tokenizer.encode_plus(
                review,
                add_special_tokens=True,  # add [CLS], [SEP]
                max_length=self._max_length,  # max length of the text that can go to BERT
                pad_to_max_length=True,  # add [PAD] tokens
                return_attention_mask=True,  # add attention mask to not focus on pad tokens
                truncation=True)
        )

    def __map_example_to_dict(self, input_ids, attention_masks, token_type_ids, label):
        return {
                   "input_ids": input_ids,
                   "token_type_ids": token_type_ids,
                   "attention_mask": attention_masks,
               }, label

    def __encode_text(self, dataframe):
        text_series = dataframe.iloc[:, 0]
        label_df = dataframe.iloc[:, 1:]
        if label_df.shape[1] != 0:
            assert self._num_of_classes == label_df.shape[
                1], f'分类数量({self._num_of_classes})与训练集标签数量({label_df.shape[1]})不一致'
        encode_data = text_series.swifter.apply(self.__convert_example_to_feature)
        bert_input = tf.data.Dataset.from_tensor_slices(
            (encode_data['input_ids'].values.tolist(),
             encode_data['attention_mask'].values.tolist(),
             encode_data['token_type_ids'].values.tolist(),
             label_df.values.tolist()
             )
        ).map(self.__map_example_to_dict)

        return bert_input

    def __train_data_pre_processing(self, data, split_rate):
        '''
        读取训练数据并进行编码

        :param data: pandas.DataFrame
        :return: train_BatchDataset, test_BatchDataset
        '''
        shuffle_data = data.sample(frac=1)
        train_data = data.iloc[:int(shuffle_data.shape[0] * split_rate)]
        encode_train_data = self.__encode_text(train_data).batch(self._batch_size)
        val_data = data.iloc[int(shuffle_data.shape[0] * split_rate):]
        encode_val_data = self.__encode_text(val_data).batch(self._batch_size)
        return encode_train_data, encode_val_data

    def __init_model(self):
        model = TFBertForMultilabelClassification.from_pretrained(self._per_train_model_dir,
                                                                  num_labels=self._num_of_classes)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate, epsilon=1e-08, clipnorm=1)
        loss = tf.keras.losses.BinaryCrossentropy()
        metric = tf.keras.metrics.CategoricalAccuracy()
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        return model

    def encode_data(self, method, data_path, split_rate=0.8):
        '''

        :param method: # 模式 ['train','predict]
        :param data_path: 文件路径, str
        :param split_rate: 训练模型训练集与验证集分割比例, float, default=0.8
        :return:
        '''
        if method == 'train':
            data_df = self.__read_data(data_path)
            if len(list(data_df.columns)[1:]) == self._num_of_classes:
                self._class_name_list = list(data_df.columns)[1:]
                print(
                    '\033[0;91m{}: I BERT/encode_data class_name_list was Updated from input data columns.\033[0m'.format(
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")))
            encode_train_data, encode_val_data = self.__train_data_pre_processing(data_df, split_rate)
            return encode_train_data, encode_val_data
        elif method == 'predict':
            data_df = self.__read_data(data_path)
            encode_test_data = self.__encode_text(data_df).batch(self._batch_size)
            return encode_test_data, data_df
        else:
            raise KeyError('load_data方法可接受“train”、“predict”两种模式加载数据')

    def load_checkpoint(self, checkpoint_path):
        '''

        :param checkpoint_path: 断点文件路径, str
        :return:
        '''
        print('\033[0;91m{}: I BERT/load_checkpoint Loading checkpoint.\033[0m'.format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")))
        self.model.load_weights(checkpoint_path)

    def train(self, encode_train_data, encode_val_data, epochs, save_weight=True):
        '''

        :param encode_train_data: 训练集, Dataset
        :param encode_val_data: 验证集, Dataset
        :param epochs: 训练次数, int
        :param save_weight: 保存权重, bool, default=True
        :return:
        '''
        print('\033[0;91m')
        self.model.fit(encode_train_data, epochs=epochs, validation_data=encode_val_data)
        print('\033[0m')

        file_name = datetime.datetime.now().strftime('BERT_%Y_%m_%d_%H_%M_%S.h5')
        if not Path(self._model_save_dir).exists():
            os.makedirs(self._model_save_dir)
        if save_weight:
            print(
                '\033[0;91m{}: I BERT/train Saving model\033[0m'.format(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")))
            self.model.save_weights(os.path.join(self._model_save_dir, file_name))

    def evaluate(self, encode_val_data, show=True, save=False):
        '''

        :param encode_val_data: 验证集, Dataset
        :param show: 输出ROC曲线图, bool, default=True
        :param save: 保存ROC曲线图, bool, default=False
        :return:
        '''
        out = self.model.predict(encode_val_data)
        labels = numpy.concatenate([label[1].numpy() for label in list(encode_val_data)])
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        if os.path.exists(r'./fonts/msyh.ttc'):
            font = FontProperties(fname=r'./fonts/msyh.ttc')
        else:
            print("\033[0;91m{}: W BERT/evaluate No such file or directory: '.\\Fonts\\msyh.ttc'.\033[0m".format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")))
            font = FontProperties()
        colors = ['#6699CC', '#666699', '#99CC66', '#FF9900']
        for i in range(self._num_of_classes):
            fpr[i], tpr[i], _ = roc_curve(labels[:, i], out[0][:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            if not show:
                print('\033[0;91m{}: I BERT/evaluate Class {} ROC area :{:0.2f}.\033[0m'.format(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), self._class_name_list[i], roc_auc[i]))
        fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), out[0].ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        all_fpr = numpy.unique(numpy.concatenate([fpr[i] for i in range(self._num_of_classes)]))
        mean_tpr = numpy.zeros_like(all_fpr)
        for i in range(self._num_of_classes):
            mean_tpr += numpy.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= self._num_of_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        for n, i in enumerate(range(0, self._num_of_classes, 4)):
            plt.figure(dpi=720)
            plt.plot(fpr["micro"], tpr["micro"],
                     label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
                     color='#FF6666', linestyle=':', linewidth=2)
            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
                     color='#CCCCCC', linestyle=':', linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
            for color, index in zip(colors, range(self._num_of_classes)[i:i + 4]):
                plt.plot(fpr[index], tpr[index],
                         label='{} ROC curve (area = {:0.2f})'.format(self._class_name_list[index][:7] + '...' if len(
                             self._class_name_list[index]) > 7 else self._class_name_list[index], roc_auc["macro"]),
                         color=color,
                         linewidth=2)
                plt.legend(loc="lower right", fontsize=5, prop=font)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic')
            if save:
                if not Path('./ROC').exists():
                    os.makedirs('./ROC')
                plt.savefig(os.path.join('./ROC', f'ROC_{n}.png'))
            if show:
                plt.show()

    def predict(self, encode_test_data, data_df=None, model_path=None, line_acc=None):
        '''

        :param encode_test_data: 测试集, Dataset
        :param data_df: 原始数据, DataFrame, default=None
        :param model_path: 加载模型权重路径, default=None
        :param line_acc: 预测结果置信准确度, float, default=None
        :return:
        '''
        if line_acc in [i / 10 for i in range(1, 10)]:
            self._line_acc = line_acc
        if model_path is not None:
            self.model.load_weights(os.path.join(model_path))
        print('\033[0;91m')
        pred_test = self.model.predict(encode_test_data, verbose=1)[0]
        print('\033[0m')
        line = numpy.array([self._line_acc for _ in range(self._num_of_classes)])
        pred_df = pandas.DataFrame([(pre > line) + 0 for pre in pred_test])
        pred_df.columns = self._class_name_list
        if data_df is not None:
            assert pred_df.shape[0] == data_df.shape[0]
            pred_df = pandas.concat([data_df, pred_df], axis=1).reset_index(drop=True)
            return pred_df
        else:
            return pred_df
