#!/usr/bin/python3n os获取文件类型
# -*- coding: utf-8 -*-
# @Time    : 2020/12/16 15:30
# @Author  : ZhouHang
# @Email   : zhouhang@idataway.com
# @File    : main.py
# @Software: PyCharm
from bert import BERT
# from args import args

if __name__ == '__main__':
    # bert_main = BERT(per_train_model_dir=args.per_train_model_dir, num_of_classes=args.num_of_classes, **vars(args))
    # if args.do_train:
    #     bert_main.train(train_data_path=args.train_data_path, epochs=args.epochs)
    # else:
    #     bert_main.predict(test_data_path=args.test_data_path, output_path=args.output_path, model_path=args.model_path)

    bert_main = BERT(per_train_model_dir='bert-base-chinese', num_of_classes=8)
    bert_main.train(train_data_path='train_data/train.xlsx', epochs=1)
    bert_main.predict(test_data_path='train_data/test.xlsx', output_path='Predict/predict.xlsx')
