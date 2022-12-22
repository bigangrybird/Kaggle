#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def main():
    ## 读取X
    lgb_pre = pd.read_csv('./output/submission_lgb.csv')
    x1 = np.array(lgb_pre['revenue'].tolist()).reshape([4398, 1])
    xgb_pre = pd.read_csv('./output/submission_xgb.csv')
    x2 = np.array(xgb_pre['revenue'].tolist()).reshape([4398,  1])
    cat_pre = pd.read_csv('./output/submission_cat.csv')
    x3 = np.array(cat_pre['revenue'].tolist()).reshape([4398, 1])
    X = np.hstack([x1, x2, x3])

    ## 读取y
    y = []
    miss = []
    i = 0
    for line in open('./input/votes.txt', 'r', encoding='utf-8').readlines()[3000:]:
        d = eval(line)
        try:
            y.append(d['revenue'])
        except:
            miss.append(i)
        i += 1
    y = np.log1p(np.array(y))

    ## 删除缺失值
    for i in miss:
        X = np.concatenate((X[0:i], X[i+1:]))
    X = np.log1p(X)

    ## model
    rr = Ridge(alpha=0, fit_intercept=True)
    rr.fit(X, y)
    pre1 = rr.predict(X)
    y_pre = rr.predict(np.log1p(np.hstack([x1, x2, x3])))
    y_pre = np.expm1(y_pre)
    print(rr.coef_)
    print(np.sqrt(mean_squared_error((pre1 + y)/2, y)))
    pre1 = (pre1 - y) * 0.85 + y
    for i in miss:
        pre1 = np.concatenate((pre1[:i], [np.log1p(y_pre[i])], pre1[i:]))
    print(pre1.shape)

    res = pd.DataFrame()
    res['id'] = range(3001, 7399)
    # res['revenue'] = y_pre
    res['revenue'] = np.expm1(pre1)
    res.to_csv('./output/submission_linear.csv', index=False)

    y1 = np.array(pd.read_csv('./output/submission_Dragon1.csv')['revenue'].tolist())
    y2 = np.array(pd.read_csv('./output/submission_Dragon2.csv')['revenue'].tolist())
    for i in miss:
        y1 = np.concatenate((y1[0:i], y1[i+1:]))
        y2 = np.concatenate((y2[0:i], y2[i+1:]))
    print(np.sqrt(mean_squared_error(np.log1p(y1), y)))
    print(np.sqrt(mean_squared_error(np.log1p(y2), y)))


if __name__ == '__main__':
    main()