#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/16 15:46
# @Author  : ZhouHang
# @Email   : zhouhang@idataway.com
# @File    : bert.py
# @Software: PyCharm
__all__ = ['BERT']

import datetime, os
import tensorflow as tf
import pandas, swifter, numpy
from pathlib import Path
from transformers import TFBertPreTrainedModel, TFBertMainLayer, BertTokenizer
from transformers.modeling_tf_utils import get_initializer


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
    def __init__(self, per_train_model_dir, num_of_classes, class_name_list=None, **kwargs):
        '''

        :param per_train_model_dir: 预训练模型保存文件夹
        :param num_of_classes: 分类数量
        :param class_name_list: 各分类对应名称
        :param kwargs:
        '''
        self._per_train_model_dir = per_train_model_dir  # 预训练模型保存文件夹
        self._num_of_classes = num_of_classes  # 分类数量
        self._split_rate = 0.8  # 训练集验证集划分比例
        self._max_length = 256  # 句子最大长度
        self._learning_rate = 2e-5  # 学习率
        self._batch_size = 64  # 每个batch的数量
        self._model_save_dir = 'ModelWeights'  # 模型权重保存路径
        self._line_acc = 0.8  # 预测结果置信准确度
        for key, value in kwargs.items():
            exec(f"self._{key}={value}")
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
        # try:
        label_df = dataframe.iloc[:, 1:]
        if label_df.shape[1] != 0:
            assert self._num_of_classes == label_df.shape[
                1], f'分类数量({self._num_of_classes})与训练集标签数量({label_df.shape[1]})不一致'

        # except:
        #     label_df = pandas.Series([])
        encode_data = text_series.swifter.apply(self.__convert_example_to_feature)
        bert_input = tf.data.Dataset.from_tensor_slices(
            (encode_data['input_ids'].values.tolist(),
             encode_data['attention_mask'].values.tolist(),
             encode_data['token_type_ids'].values.tolist(),
             label_df.values.tolist()
             )
        ).map(self.__map_example_to_dict)

        return bert_input

    def __train_data_pre_processing(self, data):
        '''
        读取训练数据并进行编码

        :param data: pandas.DataFrame
        :return: train_BatchDataset, test_BatchDataset
        '''
        shuffle_data = data.sample(frac=1)
        train_data = data.iloc[:int(shuffle_data.shape[0] * self._split_rate)]
        encode_train_data = self.__encode_text(train_data).batch(self._batch_size)
        val_data = data.iloc[int(shuffle_data.shape[0] * self._split_rate):]
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

    def train(self, train_data_path, epochs, save_weight=True, checkpoint_path=None):
        '''

        :param train_data_path:
        :param epochs:
        :param save_weight:
        :param checkpoint_path:
        :return:
        '''
        data_df = self.__read_data(train_data_path)
        encode_train_data, encode_val_data = self.__train_data_pre_processing(data_df)
        if checkpoint_path:
            print('INFO: Loading checkpoint ...')
            self.model.load_weights(checkpoint_path)
        self.model.fit(encode_train_data, epochs=epochs, validation_data=encode_val_data)
        file_name = datetime.datetime.now().strftime('BERT_%Y_%m_%d_%H_%M_%S.h5')
        if not Path(self._model_save_dir).exists():
            os.makedirs(self._model_save_dir)
        if save_weight:
            print('INFO: Saving model ...')
            self.model.save_weights(os.path.join(self._model_save_dir, file_name))
        return self.model

    def predict(self, test_data_path, output_path, model_path=None):
        '''

        :param test_data_path:
        :param output_path:
        :param model_path:
        :return:
        '''
        _, data_type = os.path.splitext(output_path)
        folder_path, _ = os.path.split(output_path)
        if data_type.lower() not in ['.csv', '.xlsx', '.xls']:
            raise NameError('不支持此类型文件')
        if not Path(folder_path).exists():
            os.makedirs(folder_path)
        if model_path is not None:
            self.model.load_weights(os.path.join(model_path))
        data_df = self.__read_data(test_data_path)
        encode_test_data = self.__encode_text(data_df).batch(self._batch_size)
        pred_test = self.model.predict(encode_test_data, verbose=1)[0]
        line = numpy.array([self._line_acc for _ in range(self._num_of_classes)])
        pred_test = pandas.DataFrame([(pre > line) + 0 for pre in pred_test])
        pred_test.columns = self._class_name_list
        assert pred_test.shape[0] == data_df.shape[0]
        pred_test = pandas.concat([data_df, pred_test], axis=1).reset_index(drop=True)
        if data_type.lower() == '.csv':
            pred_test.to_csv(output_path, encoding='utf-8-sig')
        else:
            pred_test.to_excel(output_path, index=False)
