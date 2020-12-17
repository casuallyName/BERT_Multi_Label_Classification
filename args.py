#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/16 15:33
# @Author  : zhouhang
# @Email   : zhouhang@idataway.com
# @File    : args.py
# @Software: PyCharm
import argparse

parser = argparse.ArgumentParser(description='文本分类')

parser.add_argument('--do_train', default=True, type=bool, choices=[True, False], help='训练模型')

parser.add_argument('--num_of_classes', type=int, help='分类数量')

parser.add_argument('--epoch', default=3, type=int, help='训练次数')

parser.add_argument('--train_data_path', type=str, help='训练文件保存路径')

parser.add_argument('--test_data_path', type=str, help='测试文件保存路径')

parser.add_argument('--model_path',default=None,type=str,help='加载模型路径')

parser.add_argument('--output_path', type=str, help='预测结果保存路径')

args = parser.parse_args(args=[])
