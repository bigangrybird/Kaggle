#!/usr/bin/env python
# -*- encoding: utf-8 -*-

#file: Reader.py
#time: 2019/3/10 16:24

import pandas as pd


def read_csv(filepath):
    data = pd.read_csv(filepath)
    return data


def read_text(filepath):
    res = []
    for line in open(filepath, 'r').readlines():
        line = line.rstrip()
        res.append(line)
    return res