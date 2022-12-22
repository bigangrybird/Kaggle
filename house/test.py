#!/usr/bin/env python
# -*- encoding: utf-8 -*-

#file: test.py
#time: 2019/3/14 7:51

from preprocess.Reader import read_csv


def test():
    data = read_csv("C:/Users/mrqia/Desktop/house/train.csv")
    ks = data.keys().to_list()
    print(ks)
    for line in open('C:/Users/mrqia/Desktop/house/one_hot.txt', 'r', encoding='utf8').readlines():
        line = line.rstrip()
        try:
            ks.remove(line)
        except:
            print(line)
            exit(1)
    for line in open('C:/Users/mrqia/Desktop/house/zscore.txt', 'r', encoding='utf8').readlines():
        line = line.rstrip()
        ks.remove(line)
    print(ks)


if __name__ == '__main__':
    test()