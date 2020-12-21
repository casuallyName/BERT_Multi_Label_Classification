#!/usr/bin/python3n os获取文件类型
# -*- coding: utf-8 -*-
# @Time    : 2020/12/16 15:30
# @Author  : ZhouHang
# @Email   : zhouhang@idataway.com
# @File    : main.py
# @Software: PyCharm
from bert import BERT

if __name__ == '__main__':
    # 初始化模型
    bert_main = BERT(per_train_model_dir='bert-base-chinese', num_of_classes=34, batch_size=2)
    # Encode
    # encode_train_data, encode_val_data = bert_main.encode_data(method='train', data_path='train_data/train.xlsx')
    encode_train_data, encode_val_data = bert_main.encode_data(method='train', data_path=r'D:\WorkSpace\数据\12366答对系统数据\训练数据\v10\train_Q.csv')
    # Train
    bert_main.train(encode_train_data, encode_val_data, epochs=1)
    # Evaluate
    bert_main.evaluate(encode_val_data, save=True)
    # Predict
    encode_test_data, data_df = bert_main.encode_data(method='predict', data_path='train_data/test.xlsx')
    data = bert_main.predict(encode_test_data=encode_test_data, data_df=data_df)
    print(data)
