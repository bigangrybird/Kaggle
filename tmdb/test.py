#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import pandas as pd
from tmdb.func import get_dictionary, get_json_dict, KFoldValidation, get_lgb_param, get_lgb_model, get_xgb_param, get_xgb_model, get_cgb_param, get_cgb_model

# DRAGONS
import xgboost as xgb
import lightgbm as lgb
import catboost as cat

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge

random_seed = 2019


def main():
    ## load data
    train = pd.read_csv('./input/train.csv')
    test = pd.read_csv('./input/test.csv')

    # clean data
    train.loc[train['id'] == 16, 'revenue'] = 192864
    train.loc[train['id'] == 90, 'budget'] = 30000000
    train.loc[train['id'] == 118, 'budget'] = 60000000
    train.loc[train['id'] == 149, 'budget'] = 18000000
    train.loc[train['id'] == 313, 'revenue'] = 12000000
    train.loc[train['id'] == 451, 'revenue'] = 12000000
    train.loc[train['id'] == 464, 'budget'] = 20000000
    train.loc[train['id'] == 470, 'budget'] = 13000000
    train.loc[train['id'] == 513, 'budget'] = 930000
    train.loc[train['id'] == 797, 'budget'] = 8000000
    train.loc[train['id'] == 819, 'budget'] = 90000000
    train.loc[train['id'] == 850, 'budget'] = 90000000
    train.loc[train['id'] == 1007, 'budget'] = 2
    train.loc[train['id'] == 1112, 'budget'] = 7500000
    train.loc[train['id'] == 1131, 'budget'] = 4300000
    train.loc[train['id'] == 1359, 'budget'] = 10000000
    train.loc[train['id'] == 1542, 'budget'] = 1
    train.loc[train['id'] == 1570, 'budget'] = 15800000
    train.loc[train['id'] == 1571, 'budget'] = 4000000
    train.loc[train['id'] == 1714, 'budget'] = 46000000
    train.loc[train['id'] == 1721, 'budget'] = 17500000
    train.loc[train['id'] == 1865, 'revenue'] = 25000000
    train.loc[train['id'] == 1885, 'budget'] = 12
    train.loc[train['id'] == 2091, 'budget'] = 10
    train.loc[train['id'] == 2268, 'budget'] = 17500000
    train.loc[train['id'] == 2491, 'budget'] = 6
    train.loc[train['id'] == 2602, 'budget'] = 31000000
    train.loc[train['id'] == 2612, 'budget'] = 15000000
    train.loc[train['id'] == 2696, 'budget'] = 10000000
    train.loc[train['id'] == 2801, 'budget'] = 10000000
    train.loc[train['id'] == 335, 'budget'] = 2
    train.loc[train['id'] == 348, 'budget'] = 12
    train.loc[train['id'] == 470, 'budget'] = 13000000
    train.loc[train['id'] == 513, 'budget'] = 1100000
    train.loc[train['id'] == 640, 'budget'] = 6
    train.loc[train['id'] == 696, 'budget'] = 1
    train.loc[train['id'] == 797, 'budget'] = 8000000
    train.loc[train['id'] == 850, 'budget'] = 1500000
    train.loc[train['id'] == 1199, 'budget'] = 5
    train.loc[train['id'] == 1282, 'budget'] = 9
    train.loc[train['id'] == 1347, 'budget'] = 1
    train.loc[train['id'] == 1755, 'budget'] = 2
    train.loc[train['id'] == 1801, 'budget'] = 5
    train.loc[train['id'] == 1918, 'budget'] = 592
    train.loc[train['id'] == 2033, 'budget'] = 4
    train.loc[train['id'] == 2118, 'budget'] = 344
    train.loc[train['id'] == 2252, 'budget'] = 130
    train.loc[train['id'] == 2256, 'budget'] = 1
    train.loc[train['id'] == 2696, 'budget'] = 10000000
    test.loc[test['id'] == 3033, 'budget'] = 250
    test.loc[test['id'] == 3051, 'budget'] = 50
    test.loc[test['id'] == 3084, 'budget'] = 337
    test.loc[test['id'] == 3224, 'budget'] = 4
    test.loc[test['id'] == 3594, 'budget'] = 25
    test.loc[test['id'] == 3619, 'budget'] = 500
    test.loc[test['id'] == 3831, 'budget'] = 3
    test.loc[test['id'] == 3935, 'budget'] = 500
    test.loc[test['id'] == 4049, 'budget'] = 995946
    test.loc[test['id'] == 4424, 'budget'] = 3
    test.loc[test['id'] == 4460, 'budget'] = 8
    test.loc[test['id'] == 4555, 'budget'] = 1200000
    test.loc[test['id'] == 4624, 'budget'] = 30
    test.loc[test['id'] == 4645, 'budget'] = 500
    test.loc[test['id'] == 4709, 'budget'] = 450
    test.loc[test['id'] == 4839, 'budget'] = 7
    test.loc[test['id'] == 3125, 'budget'] = 25
    test.loc[test['id'] == 3142, 'budget'] = 1
    test.loc[test['id'] == 3201, 'budget'] = 450
    test.loc[test['id'] == 3222, 'budget'] = 6
    test.loc[test['id'] == 3545, 'budget'] = 38
    test.loc[test['id'] == 3670, 'budget'] = 18
    test.loc[test['id'] == 3792, 'budget'] = 19
    test.loc[test['id'] == 3881, 'budget'] = 7
    test.loc[test['id'] == 3969, 'budget'] = 400
    test.loc[test['id'] == 4196, 'budget'] = 6
    test.loc[test['id'] == 4221, 'budget'] = 11
    test.loc[test['id'] == 4222, 'budget'] = 500
    test.loc[test['id'] == 4285, 'budget'] = 11
    test.loc[test['id'] == 4319, 'budget'] = 1
    test.loc[test['id'] == 4639, 'budget'] = 10
    test.loc[test['id'] == 4719, 'budget'] = 45
    test.loc[test['id'] == 4822, 'budget'] = 22
    test.loc[test['id'] == 4829, 'budget'] = 20
    test.loc[test['id'] == 4969, 'budget'] = 20
    test.loc[test['id'] == 5021, 'budget'] = 40
    test.loc[test['id'] == 5035, 'budget'] = 1
    test.loc[test['id'] == 5063, 'budget'] = 14
    test.loc[test['id'] == 5119, 'budget'] = 2
    test.loc[test['id'] == 5214, 'budget'] = 30
    test.loc[test['id'] == 5221, 'budget'] = 50
    test.loc[test['id'] == 4903, 'budget'] = 15
    test.loc[test['id'] == 4983, 'budget'] = 3
    test.loc[test['id'] == 5102, 'budget'] = 28
    test.loc[test['id'] == 5217, 'budget'] = 75
    test.loc[test['id'] == 5224, 'budget'] = 3
    test.loc[test['id'] == 5469, 'budget'] = 20
    test.loc[test['id'] == 5840, 'budget'] = 1
    test.loc[test['id'] == 5960, 'budget'] = 30
    test.loc[test['id'] == 6506, 'budget'] = 11
    test.loc[test['id'] == 6553, 'budget'] = 280
    test.loc[test['id'] == 6561, 'budget'] = 7
    test.loc[test['id'] == 6582, 'budget'] = 218
    test.loc[test['id'] == 6638, 'budget'] = 5
    test.loc[test['id'] == 6749, 'budget'] = 8
    test.loc[test['id'] == 6759, 'budget'] = 50
    test.loc[test['id'] == 6856, 'budget'] = 10
    test.loc[test['id'] == 6858, 'budget'] = 100
    test.loc[test['id'] == 6876, 'budget'] = 250
    test.loc[test['id'] == 6972, 'budget'] = 1
    test.loc[test['id'] == 7079, 'budget'] = 8000000
    test.loc[test['id'] == 7150, 'budget'] = 118
    test.loc[test['id'] == 6506, 'budget'] = 118
    test.loc[test['id'] == 7225, 'budget'] = 6
    test.loc[test['id'] == 7231, 'budget'] = 85
    test.loc[test['id'] == 5222, 'budget'] = 5
    test.loc[test['id'] == 5322, 'budget'] = 90
    test.loc[test['id'] == 5350, 'budget'] = 70
    test.loc[test['id'] == 5378, 'budget'] = 10
    test.loc[test['id'] == 5545, 'budget'] = 80
    test.loc[test['id'] == 5810, 'budget'] = 8
    test.loc[test['id'] == 5926, 'budget'] = 300
    test.loc[test['id'] == 5927, 'budget'] = 4
    test.loc[test['id'] == 5986, 'budget'] = 1
    test.loc[test['id'] == 6053, 'budget'] = 20
    test.loc[test['id'] == 6104, 'budget'] = 1
    test.loc[test['id'] == 6130, 'budget'] = 30
    test.loc[test['id'] == 6301, 'budget'] = 150
    test.loc[test['id'] == 6276, 'budget'] = 100
    test.loc[test['id'] == 6473, 'budget'] = 100
    test.loc[test['id'] == 6842, 'budget'] = 30

    # external data
    release_dates = pd.read_csv('./input/release_dates_per_country.csv')
    release_dates['id'] = range(1, 7399)
    release_dates.drop(['original_title', 'title'], axis=1, inplace=True)
    train = pd.merge(train, release_dates, how='left', on=['id'])
    test = pd.merge(test, release_dates, how='left', on=['id'])
    vote = pd.read_csv('./input/votes.csv')
    train = pd.merge(train, vote, how='left', on=['id'])
    test = pd.merge(test, vote, how='left', on=['id'])
    trainAdditionalFeatures = pd.read_csv('./input/TrainAdditionalFeatures.csv')[
        ['imdb_id', 'popularity2', 'rating']]
    testAdditionalFeatures = pd.read_csv('./input/TestAdditionalFeatures.csv')[
        ['imdb_id', 'popularity2', 'rating']]
    train = pd.merge(train, trainAdditionalFeatures, how='left', on=['imdb_id'])
    test = pd.merge(test, testAdditionalFeatures, how='left', on=['imdb_id'])


    test['revenue'] = np.nan
    json_cols = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'Keywords', 'cast',
                 'crew']

    ## 将json形式的feature值从字符串转换成json
    for col in json_cols + ['belongs_to_collection']:
        train[col] = train[col].apply(lambda x: get_dictionary(x))
        test[col] = test[col].apply(lambda x: get_dictionary(x))

    ## 获得json_cols里面的key以及计数
    train_dict = get_json_dict(train, json_cols)
    test_dict = get_json_dict(test, json_cols)


    for col in json_cols:
        remove = []
        train_id = set(list(train_dict[col].keys()))
        test_id = set(list(test_dict[col].keys()))

        remove += list(train_id - test_id) + list(test_id - train_id)
        for i in train_id.union(test_id) - set(remove):
            if train_dict[col][i] < 5 or i == '':
                remove += [i]

        for i in remove:
            if i in train_dict[col]:
                del train_dict[col][i]
            if i in test_dict[col]:
                del test_dict[col][i]

        # print(col, 'size :', len(train_id.union(test_id)), '->', len(train_dict[col]))
    df = pd.concat([train, test], sort=True).reset_index(drop=True)
    df[['release_month', 'release_day', 'release_year']] = df['release_date'].str.split('/', expand=True).replace(
        np.nan, 0).astype(int)
    df.loc[(df['release_year'] <= 18) & (df['release_year'] < 100), "release_year"] += 2000
    df.loc[(df['release_year'] > 18) & (df['release_year'] < 100), "release_year"] += 1900
    rating_na = df.groupby(["release_year", "original_language"])['rating'].mean().reset_index()
    df.loc[df.rating.isna(), 'rating'] = df.merge(rating_na, how='left', on=["release_year", "original_language"])
    vote_count_na = df.groupby(["release_year", "original_language"])['vote_count'].mean().reset_index()
    df.loc[df.vote_count.isna(), 'vote_count'] = df.merge(vote_count_na, how='left',
                                                      on=["release_year", "original_language"])
    budget_na = df.groupby(["release_year","original_language"])['budget'].mean().reset_index()
    df.loc[df.budget == 0, 'budget'] = df.merge(budget_na, how = 'left' ,on = ["release_year","original_language"])
    df['budget'] = np.log1p(df['budget'])
    df['genders_0_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
    df['genders_1_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
    df['genders_2_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))
    df['_collection_name'] = df['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)
    le = LabelEncoder()
    le.fit(list(df['_collection_name'].fillna('')))
    df['_collection_name'] = le.transform(df['_collection_name'].fillna('').astype(str))
    df['_num_Keywords'] = df['Keywords'].apply(lambda x: len(x) if x != {} else 0)
    df['_num_cast'] = df['cast'].apply(lambda x: len(x) if x != {} else 0)
    releaseDate = pd.to_datetime(df['release_date'])
    df['release_dayofweek'] = releaseDate.dt.dayofweek
    df['release_quarter'] = releaseDate.dt.quarter
    df['_budget_runtime_ratio'] = df['budget'] / df['runtime']
    df['_budget_popularity_ratio'] = df['budget'] / df['popularity']
    df['_budget_year_ratio'] = df['budget'] / (df['release_year'] * df['release_year'])
    df['_releaseYear_popularity_ratio'] = df['release_year'] / df['popularity']
    df['_releaseYear_popularity_ratio2'] = df['popularity'] / df['release_year']
    df['meanruntimeByYear'] = df.groupby("release_year")["runtime"].aggregate('mean')
    df['meanPopularityByYear'] = df.groupby("release_year")["popularity"].aggregate('mean')
    df['meanBudgetByYear'] = df.groupby("release_year")["budget"].aggregate('mean')
    df['_popularity_theatrical_ratio'] = df['theatrical'] / df['popularity']
    df['_budget_theatrical_ratio'] = df['budget'] / df['theatrical']
    df['mean_theatrical_ByYear'] = df.groupby("release_year")["theatrical"].aggregate('mean')
    df['_popularity_totalVotes_ratio'] = df['vote_count'] / df['popularity']
    df['_totalVotes_releaseYear_ratio'] = df['vote_count'] / df['release_year']
    df['_budget_totalVotes_ratio'] = df['budget'] / df['vote_count']
    df['_rating_popularity_ratio'] = df['rating'] / df['popularity']
    df['_rating_totalVotes_ratio'] = df['vote_count'] / df['rating']
    df['_budget_rating_ratio'] = df['budget'] / df['rating']
    df['_runtime_rating_ratio'] = df['runtime'] / df['rating']
    df['has_homepage'] = 0
    df.loc[pd.isnull(df['homepage']), "has_homepage"] = 1
    df['isbelongs_to_collectionNA'] = 0
    df.loc[pd.isnull(df['belongs_to_collection']), "isbelongs_to_collectionNA"] = 1
    df['isTaglineNA'] = 0
    df.loc[df['tagline'] == 0, "isTaglineNA"] = 1
    df['isOriginalLanguageEng'] = 0
    df.loc[df['original_language'] == "en", "isOriginalLanguageEng"] = 1
    df['isTitleDifferent'] = 1
    df.loc[df['original_title'] == df['title'], "isTitleDifferent"] = 0
    df['isMovieReleased'] = 1
    df.loc[df['status'] != "Released", "isMovieReleased"] = 0
    df['collection_id'] = df['belongs_to_collection'].apply(lambda x: np.nan if len(x) == 0 else x[0]['id'])
    df['original_title_letter_count'] = df['original_title'].str.len()
    df['original_title_word_count'] = df['original_title'].str.split().str.len()
    df['title_word_count'] = df['title'].str.split().str.len()
    df['overview_word_count'] = df['overview'].str.split().str.len()
    df['tagline_word_count'] = df['tagline'].str.split().str.len()
    df['production_countries_count'] = df['production_countries'].apply(lambda x: len(x))
    df['production_companies_count'] = df['production_companies'].apply(lambda x: len(x))
    df['cast_count'] = df['cast'].apply(lambda x: len(x))
    df['crew_count'] = df['crew'].apply(lambda x: len(x))
    # for col in ['genres', 'production_countries', 'spoken_languages', 'production_companies', 'Keywords', 'cast']:
    for col in json_cols:
        df[col] = df[col].map(lambda x: sorted(list(set([n if n in train_dict[col] else col + '_etc' for n in [d['name'] for d in x]])))).map(lambda x: ','.join(map(str, x)))
        temp = df[col].str.get_dummies(sep=',')
        temp2 = pd.DataFrame()
        for k in temp.keys():
            temp2[col + '_' + k] = temp[k]
        df = pd.concat([df, temp2], axis=1, sort=False)
    df.drop(['genres_genres_etc'], axis=1, inplace=True)
    df = df.drop(['belongs_to_collection', 'genres', 'homepage', 'imdb_id', 'overview', 'runtime'
                     , 'poster_path', 'production_companies', 'production_countries', 'release_date',
                  'spoken_languages'
                     , 'status', 'title', 'Keywords', 'cast', 'crew', 'original_language', 'original_title',
                  'tagline', 'collection_id', 'movie_id'
                  ], axis=1)
    df.fillna(value=0.0, inplace=True)
    train = df.loc[:train.shape[0] - 1, :]
    test = df.loc[train.shape[0]:, :]
    features = list(train.columns)
    features = [i for i in features if i != 'id' and i != 'revenue']
    print(train.shape)
    # print(len(train.keys()))
    # print(len(set(train.keys())))
    # d = dict()
    # for k in train.keys():
    #     d[k] = d.get(k, 0) + 1
    # for k, v in d.items():
    #     if v > 1:
    #         print(k)
    # print(col)
    # # print(df[col])
    # print(temp)

    ## Model
    Kfolder = KFoldValidation(train)
    # [p1, p2, p3, p4, p5, p6, p7, p8] = get_lgb_param(train, test, features, 3)
    # lgbmodel = get_lgb_model(p1, p2, p3, p4, p5, p6, p7, p8)

    # lgbmodel = lgb.LGBMRegressor(n_estimators=10000,
    #                              objective='regression',
    #                              metric='rmse',
    #                              max_depth=5,
    #                              num_leaves=30,
    #                              min_child_samples=100,
    #                              learning_rate=0.01,
    #                              boosting='rf',
    #                              feature_fraction=0.9,
    #                              bagging_freq=1,
    #                              bagging_fraction=0.9,
    #                              importance_type='gain',
    #                              lambda_l1=0.2,
    #                              bagging_seed=random_seed,
    #                              subsample=.8,
    #                              colsample_bytree=.9,
    #                              use_best_model=True)
    # Kfolder.validate(train, test, features, lgbmodel, name="lgbfinal", prepare_stacking=True)
    #
    # xgbmodel = xgb.XGBRegressor(max_depth=5,
    #                             learning_rate=0.01,
    #                             n_estimators=10000,
    #                             objective='reg:linear',
    #                             gamma=1.45,
    #                             seed=random_seed,
    #                             silent=True,
    #                             subsample=0.8,
    #                             colsample_bytree=0.7,
    #                             colsample_bylevel=0.5)
    # [p1, p2, p3, p4, p5, p6, p7] = get_xgb_param(train, test, features, 3)
    # xgbmodel = get_xgb_model(p1, p2, p3, p4, p5, p6, p7)
    # Kfolder.validate(train, test, features, xgbmodel, name="xgbfinal", prepare_stacking=True)
    # catmodel = cat.CatBoostRegressor(iterations=10000,
    #                                  learning_rate=0.01,
    #                                  depth=5,
    #                                  eval_metric='RMSE',
    #                                  colsample_bylevel=0.8,
    #                                  bagging_temperature=0.2,
    #                                  metric_period=None,
    #                                  early_stopping_rounds=200,
    #                                  random_seed=random_seed)
    [p1, p2, p3, p4, p5] = get_cgb_param(train, test, features, 3)
    catmodel = get_cgb_model(p1, p2, p3, p4, p5)
    Kfolder.validate(train, test, features, catmodel, name="catfinal", prepare_stacking=True,
                     fit_params={"use_best_model": True, "verbose": 100})

    # test['revenue'] = np.expm1(test["lgbfinal"])
    # test[['id', 'revenue']].to_csv('./output/submission_lgb.csv', index=False)
    # test['revenue'] = np.expm1(test["xgbfinal"])
    # test[['id', 'revenue']].to_csv('./output/submission_xgb.csv', index=False)
    # test['revenue'] = np.expm1(test["catfinal"])
    # test[['id', 'revenue']].to_csv('./output/submission_cat.csv', index=False)
    # test['revenue'] = np.expm1(0.4 * test["lgbfinal"] + 0.4 * test["catfinal"] + 0.2 * test["xgbfinal"])
    # test[['id', 'revenue']].to_csv('./output/submission_Dragon1.csv', index=False)
    # test['revenue'] = np.expm1((test["lgbfinal"] + test["catfinal"] + test["xgbfinal"]) / 3)
    # test[['id', 'revenue']].to_csv('./output/submission_Dragon2.csv', index=False)

    # rr = Ridge(alpha=0.01)
    # X = pd.concat([test['lgbfinal'], test['xgbfinal'], test['catfinal']])


if __name__ == '__main__':
    main()