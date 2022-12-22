#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cat


def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d


def get_json_dict(df, json_cols):
    result = dict()
    for e_col in json_cols:
        d = dict()
        rows = df[e_col].values
        for row in rows:
            if row is None: continue
            if len(row) == 0: continue
            for i in row[0:5]:
                if i['name'] not in d:
                    d[i['name']] = 0
                d[i['name']] += 1
        result[e_col] = d
    return result


def score(data, y):
    validation_res = pd.DataFrame(
        {"id": data["id"].values,
         "transactionrevenue": data["revenue"].values,
         "predictedrevenue": np.expm1(y)})
    validation_res = validation_res.groupby("id")["transactionrevenue", "predictedrevenue"].sum().reset_index()
    return np.sqrt(mean_squared_error(np.log1p(validation_res["transactionrevenue"].values),
                                      np.log1p(validation_res["predictedrevenue"].values)))


class KFoldValidation():
    def __init__(self, data, n_splits=5):
        unique_vis = np.array(sorted(data['id'].astype(str).unique()))
        folds = GroupKFold(n_splits)
        ids = np.arange(data.shape[0])

        self.fold_ids = []
        for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
            self.fold_ids.append([
                ids[data['id'].astype(str).isin(unique_vis[trn_vis])],
                ids[data['id'].astype(str).isin(unique_vis[val_vis])]
            ])

    def validate(self, train, test, features, model, name="", prepare_stacking=False,
                 fit_params={"early_stopping_rounds": 500, "verbose": 100, "eval_metric": "rmse"}):
        model.FI = pd.DataFrame(index=features)
        full_score = 0

        if prepare_stacking:
            test[name] = 0
            train[name] = np.NaN

        for fold_id, (trn, val) in enumerate(self.fold_ids):
            devel = train[features].iloc[trn]
            y_devel = np.log1p(train["revenue"].iloc[trn])
            valid = train[features].iloc[val]
            y_valid = np.log1p(train["revenue"].iloc[val])

            print("Fold ", fold_id, ":")
            model.fit(devel, y_devel, eval_set=[(valid, y_valid)], **fit_params)

            if len(model.feature_importances_) == len(features):
                model.FI['fold' + str(fold_id)] = model.feature_importances_ / model.feature_importances_.sum()

            predictions = model.predict(valid)
            predictions[predictions < 0] = 0
            print("Fold ", fold_id, " error: ", mean_squared_error(y_valid, predictions) ** 0.5)

            fold_score = score(train.iloc[val], predictions)
            full_score += fold_score / len(self.fold_ids)
            print("Fold ", fold_id, " score: ", fold_score)
            if prepare_stacking:
                train[name].iloc[val] = predictions

                test_predictions = model.predict(test[features])
                test_predictions[test_predictions < 0] = 0
                test[name] += test_predictions / len(self.fold_ids)

        print("Final score: ", full_score)
        return full_score


def get_lgb_model(p1, p2, p3, p4, p5, p6, p7, p8):
    return lgb.LGBMRegressor(n_estimators=p1,
                             learning_rate=p2,
                             objective='regression',
                              metric='rmse',
                              max_depth=p3,
                              num_leaves=p4,
                              min_child_samples=p5,
                              boosting='gbdt',
                              feature_fraction=p6,
                              bagging_freq=1,
                              bagging_fraction=p7,
                              importance_type='gain',
                              lambda_l1=p8,
                              bagging_seed=2019,
                              subsample=.8,
                              colsample_bytree=.9,
                              use_best_model=True)


def get_lgb_param(train, test, features, epochs):
    ## 候选参数
    lgb_paramss = {
        'n_estimators': [2000, 3000, 4000, 5000, 6000],
        'learning_rate': [1, 0.3, 0.1, 0.03, 0.01],
        'max_depth': [4, 5, 6, 7],
        'num_leaves': [25, 30, 35, 40],
        'min_child_samples': [60, 80, 100, 120],
        'feature_fracation': [0.6, 0.7, 0.8, 0.9],
        'bagging_fraction': [0.8, 0.9, 1.0],
        'lambda_l1': [0.2, 0.3, 0.4]
    }
    ## 基准参数
    lgb_params = {
        'n_estimators': 3000,
        'learning_rate': 0.01,
        'max_depth': 5,
        'num_leaves': 30,
        'min_child_samples': 100,
        'feature_fracation': 0.9,
        'bagging_fraction': 0.9,
        'lambda_l1': 0.2
    }
    res = []
    ## 尝试所有参数
    kFold = KFoldValidation(train)
    for _ in range(epochs):
        for para, values in lgb_paramss.items():
            tmp = list()
            for value in values:
                lgb_params[para] = value
                [p1, p2, p3, p4, p5, p6, p7, p8] = list(lgb_params.values())
                model = get_lgb_model(p1, p2, p3, p4, p5, p6, p7, p8)
                score = kFold.validate(train, test, features, model)
                tmp.append({'score': score, 'params':[p1, p2, p3, p4, p5, p6, p7, p8]})
            ## 找出当前尝试的参数中效果最好的
            min_score = 1000000.0
            paras = []
            for d in tmp:
                if d['score'] < min_score:
                    min_score = d['score']
                    paras = d['params']
            ## 更新基准参数
            lgb_params = {
                'n_estimators': paras[0],
                'learning_rate': paras[1],
                'max_depth': paras[2],
                'num_leaves': paras[3],
                'min_child_samples': paras[4],
                'feature_fracation': paras[5],
                'bagging_fraction': paras[6],
                'lambda_l1': paras[7]
            }
            res.append(tmp)

    f = open('./output/lgb_params.log', 'w')
    for l in res:
        for d in l:
            f.write('\t'.join([str(x) for x in d['params']]))
            f.write('\t' + str(d['score']) + '\n')
    f.close()
    return paras


def get_xgb_model(p1, p2, p3, p4, p5, p6, p7):
    return xgb.XGBRegressor(n_estimators=p1,
                            learning_rate=p2,
                            max_depth=p3,
                            objective='reg:linear',
                            gamma=p4,
                            seed=2019,
                            silent=True,
                            subsample=p5,
                            colsample_bytree=p6,
                            colsample_bylevel=p7)



def get_xgb_param(train, test, features, epochs):
    ## 候选参数
    xgb_paramss = {
        'n_estimators': [2000, 3000, 4000, 5000, 6000],
        'learning_rate': [0.3, 0.1, 0.03, 0.01, 0.003],
        'max_depth': [4, 5, 6, 7],
        'gamma': [0.3, 0.6, 1, 1.3, 1.6],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
        'colsample_bylevel': [0.3, 0.4, 0.5, 0.6]
    }
    ## 基准参数
    xgb_params = {
        'n_estimators': 3000,
        'learning_rate': 0.01,
        'max_depth': 5,
        'gamma': 1.45,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 0.5,
    }
    res = []
    ## 尝试所有参数
    kFold = KFoldValidation(train)
    for _ in range(epochs):
        for para, values in xgb_paramss.items():
            tmp = list()
            for value in values:
                xgb_params[para] = value
                [p1, p2, p3, p4, p5, p6, p7] = list(xgb_params.values())
                model = get_xgb_model(p1, p2, p3, p4, p5, p6, p7)
                score = kFold.validate(train, test, features, model)
                tmp.append({'score': score, 'params': [p1, p2, p3, p4, p5, p6, p7]})
            ## 找出当前尝试的参数中效果最好的
            min_score = 1000000.0
            paras = []
            for d in tmp:
                if d['score'] < min_score:
                    min_score = d['score']
                    paras = d['params']
            ## 更新基准参数
            xgb_params = {
                        'n_estimators': paras[0],
                        'learning_rate': paras[1],
                        'max_depth': paras[2],
                        'gamma': paras[3],
                        'subsample': paras[4],
                        'colsample_bytree': paras[5],
                        'colsample_bylevel': paras[6],
                        }
            res.append(tmp)

    f = open('./output/xgb_params.log', 'w')
    for l in res:
        for d in l:
            f.write('\t'.join([str(x) for x in d['params']]))
            f.write('\t' + str(d['score']) + '\n')
    f.close()
    return paras


def get_cgb_model(p1, p2, p3, p4, p5):
    return cat.CatBoostRegressor(iterations=p1,
                                     learning_rate=p2,
                                     depth=p3,
                                     eval_metric='RMSE',
                                     colsample_bylevel=p4,
                                     bagging_temperature=p5,
                                     metric_period=None,
                                     early_stopping_rounds=500,
                                     random_seed=2019)



def get_cgb_param(train, test, features, epochs):
    ## 候选参数
    cgb_paramss = {
        'iterations': [2000, 3000, 4000, 5000, 6000],
        'learning_rate': [0.3, 0.1, 0.03, 0.01, 0.003],
        'max_depth': [3, 4, 5, 6, 7],
        'colsample_bylevel': [0.7, 0.8, 0.9, 1.0],
        'bagging_temperature': [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    ## 基准参数
    cgb_params = {
        'iterations': 3000,
        'learning_rate': 0.01,
        'max_depth': 5,
        'colsample_bylevel': 0.8,
        'bagging_temperature': 0.2,
    }
    res = []
    ## 尝试所有参数
    kFold = KFoldValidation(train)
    for _ in range(epochs):
        for para, values in cgb_paramss.items():
            tmp = list()
            for value in values:
                cgb_params[para] = value
                [p1, p2, p3, p4, p5] = list(cgb_params.values())
                model = get_cgb_model(p1, p2, p3, p4, p5)
                score = kFold.validate(train, test, features, model, fit_params={"use_best_model": True, "verbose": 100})
                tmp.append({'score': score, 'params': [p1, p2, p3, p4, p5]})
            ## 找出当前尝试的参数中效果最好的
            min_score = 1000000.0
            paras = []
            for d in tmp:
                if d['score'] < min_score:
                    min_score = d['score']
                    paras = d['params']
            ## 更新基准参数
            cgb_params = {
                'iterations': paras[0],
                'learning_rate': paras[1],
                'max_depth': paras[2],
                'colsample_bylevel': paras[3],
                'bagging_temperature': paras[4],
            }
            res.append(tmp)

    f = open('./output/cgb_params.log', 'w')
    for l in res:
        for d in l:
            f.write('\t'.join([str(x) for x in d['params']]))
            f.write('\t' + str(d['score']) + '\n')
    f.close()
    return paras