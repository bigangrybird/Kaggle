#!/usr/bin/env python
# -*- encoding: utf-8 -*-

#file: EDA.py
#time: 2019/3/10 18:18

import math
import numpy

def show_value_count(data, feature, num=True):
    """
    本方法用于统计feature每种值的数量，其实主要是统计缺失值的数量
    :param data: 原始特征数据
    :param feature: 要统计的特征
    :return:
    """
    ## 统计数目
    d = dict()
    for val in data[feature]:
        if num:
            if math.isnan(val):
                val = 'nan'
        d[val] = d.get(val, 0) + 1

    ## 按count倒序打印出各个值
    for k, v in sorted(d.items(), key = lambda x:x[1] , reverse=True):
        print(k, v)
    print(sum(d.values()))


def one_hot(data, feature):
    """
    本方法用于将原始特征数据中的某一个特征数据做one-hot编码
    :param data: 原始特征数据
    :param feature: 需要做one-hot编码的特征，根据原始特征数据的类型，feature可以是index（int）或者key（str）
    :return:
    """
    ## 将特征值提取出来，并计算one-hot向量的长度
    feature_vals = data[feature]
    codes = list(set(feature_vals))
    length = len(codes)

    ## 将每个特征值都转换成相应的one-hot向量
    for i in range(len(feature_vals)):
        temp_val = [0] * length
        temp_val[codes.index(feature_vals[i])] = 1
        feature_vals[i] = temp_val


def correlation_coefficent(data, feature, res):
    """
    本方法用于计算有缺失值的特征和结果之间的相关系数，帮助判断是否可以舍弃该特征
    :param data: 原始特征数据
    :param feature: 拥有缺失值的特征
    :param res: 结果
    :return:
    """
    feature_vals = data[feature]
    res_vals = data[res]
    a = []
    b = []
    for i in range(len(feature_vals)):
        if math.isnan(feature_vals[i]):
            continue
        a.append(feature_vals[i])
        b.append(res_vals[i])

    pc = numpy.corrcoef(a, b)
    print(pc)


def filling(data, feature, cat='mean'):
    """
    本方法用于填充缺失值的特征。‘mean’模式给缺失值填充平均值，
    ‘mode’模式给缺失值填充众数,
    ‘zero'模式给缺失值填充0.
    :param data: 原始特征数据
    :param feature: 要进行填充的特征
    :param cat: 填充的模式
    :return:
    """
    feature_vals = data[feature]
    miss = []
    if cat == 'mean':
        ## 如果是填充平均值
        sum_vals = 0
        count = 0
        for i in range(len(feature_vals)):
            if math.isnan(feature_vals[i]):
                miss.append(i)
            else:
                sum_vals += feature_vals[i]
                count += 1
        for j in miss:
            m = sum_vals * 1.0 / count
            feature_vals[j] = m
    elif cat == 'mode':
        ## 如果是填充众数
        max_val = 0
        max_count = 0
        d = dict()
        for i in range(len(feature_vals)):
            if isinstance(feature_vals[i], float) and math.isnan(feature_vals[i]):
                miss.append(i)
            else:
                d[feature_vals[i]] = d.get(feature_vals[i], 0) + 1
                if d[feature_vals[i]] > max_count:
                    max_val = feature_vals[i]
                    max_count = d[feature_vals[i]]
        for j in miss:
            feature_vals[j] = max_val
    elif cat == 'zero':
        for i in range(len(feature_vals)):
            if math.isnan(feature_vals[i]):
                feature_vals[i] = 0
    else:
        print("Only two modes of filling are supported, 'mean' and 'mode', and you choose %s." % cat)
        exit(1)


def zscore(data, feature):
    """
    将特征值进行标准化
    :param data: 原始数据集合
    :param feature: 需要进行标准化的特征
    :return:
    """
    feature_vals = data[feature]
    m = numpy.mean(feature_vals)
    std = numpy.std(feature_vals, ddof=1)
    # if math.isnan(m):
    #     print(feature, feature_vals)
    # print(max(feature_vals), min(feature_vals), m, std)
    data[feature] = (feature_vals - m) / std

    ## 附加一个异常值检测功能
    for val in feature_vals:
        if numpy.abs(val - m) > 9 * std:
            print(feature)
            break