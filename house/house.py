#!/usr/bin/env python
# -*- encoding: utf-8 -*-

#file: house.py
#time: 2019/3/11 7:33

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, skew
from preprocess.Reader import *
from preprocess.EDA import *
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


def read():
    basedir = 'C:/Users/mrqia/Desktop/house/'
    train_filepath = basedir + "train.csv"
    test_filepath = basedir + "test.csv"
    train = read_csv(train_filepath)
    test = read_csv(test_filepath)

    # 存储ID
    train_ID = train['Id']
    test_ID = test['Id']
    # 在数据中删除ID
    train.drop('Id', axis=1, inplace=True)
    test.drop('Id', axis=1, inplace=True)
    return train, test, test_ID


# def explore_data():
#     train_data, test_data, train_len, test_len = read()
#     features = train_data.keys()[1: -1].to_list()
#     raw_data = dict()
#     for f in features:
#         raw_data[f] = train_data[f].to_list() + test_data[f].to_list()
#         i = 0
#         for val in raw_data[f]:
#             i += 1
#             if isinstance(val, float):
#                 if math.isnan(val):
#                     print(f, i)
#                     break


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def preprocess():
    """
    本方法为数据预处理部分，做诸如补充缺失值、做类别特征编码、标准化等操作
    :return:
    """
    ## 读取数据
    train, test, test_ID = read()

    ## 处理outlier
    train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

    ## 分析target variable，用log变换将其调整成接近高斯分布的状态
    train['SalePrice'] = np.log1p(train['SalePrice'])

    ## 特征工程
    ntrain = train.shape[0]
    ntest = test.shape[0]
    y_train = train.SalePrice.values
    all_data = pd.concat((train, test), sort=True).reset_index(drop=True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)

    ## 处理缺失值
    all_data['PoolQC'] = all_data['PoolQC'].fillna('None')
    all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
    all_data["Alley"] = all_data["Alley"].fillna("None")
    all_data["Fence"] = all_data["Fence"].fillna("None")
    all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        all_data[col] = all_data[col].fillna('None')
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all_data[col] = all_data[col].fillna(0)
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[col] = all_data[col].fillna(0)
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        all_data[col] = all_data[col].fillna('None')
    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
    all_data = all_data.drop(['Utilities'], axis=1)
    all_data["Functional"] = all_data["Functional"].fillna("Typ")
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
    all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

    ## 把一些应该是类别型特征从数字表示转换为字符串表示
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)

    ## 增加两个特征
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    all_data['Remodeled'] = all_data['YearBuilt'] == all_data['YearRemodAdd']
    all_data['Remodeled'] = all_data['Remodeled'].astype(str)

    from sklearn.preprocessing import LabelEncoder
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
            'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
            'YrSold', 'MoSold', 'Remodeled')
    # process columns, apply LabelEncoder to categorical features
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(all_data[c].values))
        all_data[c] = lbl.transform(list(all_data[c].values))
    # shape
    # print('Shape all_data: {}'.format(all_data.shape))

    ## Skewed features
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_feats})
    skewness = skewness[abs(skewness) > 0.75]
    from scipy.special import boxcox1p
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        all_data[feat] = boxcox1p(all_data[feat], lam)
    #
    # ## dummy features
    all_data = pd.get_dummies(all_data)
    # print(all_data.shape)
    train = all_data[:ntrain]
    test = all_data[ntrain:]
    #
    # ## 建模
    n_folds = 5
    def rmsle_cv(model):
        kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
        rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
        return rmse

    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3))
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='huber', random_state=5)
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                                 learning_rate=0.05, max_depth=3,
                                 min_child_weight=1.7817, n_estimators=2400,
                                 reg_alpha=0.4640, reg_lambda=0.8571,
                                 subsample=0.5213, silent=True,
                                 random_state=7, nthread=-1)
    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                  learning_rate=0.05, n_estimators=800,
                                  max_bin=55, bagging_fraction=0.8,
                                  bagging_freq=5, feature_fraction=0.2319,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
    averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))
    stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR),
                                                     meta_model=lasso)
    rf = RandomForestRegressor(n_estimators=3000, max_depth=5, min_samples_split=10,
                               min_samples_leaf=15, random_state=5, oob_score=True)

    # for para in [2, 4, 6, 8, 10, 12]:
    #     rf = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_split=para,
    #                                min_samples_leaf=2, random_state=5, oob_score=True)
    #     score = rmsle_cv(rf)
    #     print('RF: %.8f error with %.8f std with %.2f.' % (score.mean(), score.std(), para))
    # score = rmsle_cv(lasso)
    # print('Lasso_CV: %.8f error with %.8f std.' % (score.mean(), score.std()))
    # score = rmsle_cv(ENet)
    # print('ENet: %.8f error with %.8f std.' % (score.mean(), score.std()))
    # score = rmsle_cv(KRR)
    # print('KRR: %.8f error with %.8f std.' % (score.mean(), score.std()))
    # score = rmsle_cv(GBoost)
    # print('GBoost: %.8f error with %.8f std.' % (score.mean(), score.std()))
    # score = rmsle_cv(model_xgb)
    # print('XGB: %.8f error with %.8f std.' % (score.mean(), score.std()))
    # score = rmsle_cv(model_lgb)
    # print('LGB: %.8f error with %.8f std.' % (score.mean(), score.std()))
    # score = rmsle_cv(averaged_models)
    # print(" Averaged base models score: {:.8f} ({:.8f})\n".format(score.mean(), score.std()))
    # score = rmsle_cv(stacked_averaged_models)
    # print("Stacking Averaged models score: {:.8f} ({:.8f})".format(score.mean(), score.std()))
    # score = rmsle_cv(rf)
    # print('RF: %.8f error with %.8f std.' % (score.mean(), score.std()))

    ## 平均stacking
    stacked_averaged_models.fit(train.values, y_train)
    stacked_train_pred = stacked_averaged_models.predict(train.values)
    stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
    print('Stacked_ALL: %.8f' % rmsle(y_train, stacked_train_pred))
    # xgb
    model_xgb.fit(train, y_train)
    xgb_train_pred = model_xgb.predict(train)
    xgb_pred = np.expm1(model_xgb.predict(test))
    print('XGB_ALL: %.8f' % rmsle(y_train, xgb_train_pred))
    # # lgb
    model_lgb.fit(train, y_train)
    lgb_train_pred = model_lgb.predict(train)
    lgb_pred = np.expm1(model_lgb.predict(test.values))
    print('LGB_ALL: %.8f' % rmsle(y_train, lgb_train_pred))
    print('RMSLE score on train data:')
    print(rmsle(y_train, stacked_train_pred * 0.50 +
                xgb_train_pred * 0.10 + lgb_train_pred * 0.40))

    ## 制作提交文件
    ensemble = stacked_pred * 0.50 + xgb_pred * 0.10 + lgb_pred * 0.40
    sub = pd.DataFrame()
    sub['Id'] = test_ID
    sub['SalePrice'] = ensemble
    sub.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    preprocess()