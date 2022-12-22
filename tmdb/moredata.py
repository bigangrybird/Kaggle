#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import requests
import pandas as pd
import time
import math


baseurl = "https://z4vrpkijmodhwsxzc.stoplight-proxy.io/3/movie/"
apikey =  "6e839191f9e801d8346ac2f8e8a0eb56"
language = "en-US"


def download():
    train = pd.read_csv('./input/train.csv')
    test = pd.read_csv('./input/test.csv')
    imdb_ids = train['imdb_id'].to_list() + test['imdb_id'].to_list()

    f = open('./input/votes.txt', 'a', encoding='utf-8')
    for imdb_id in imdb_ids[7213:]:
        url = baseurl + imdb_id + "?api_key=" + apikey + "&language=" + language
        r = requests.get(url).json()
        f.write(str(r) + '\n')
        f.flush()
        time.sleep(2)
    f.close()


def rewrite():
    vote_count = []
    vote_average = []
    i = 0
    for line in open('./input/votes.txt', 'r', encoding='utf-8').readlines():
        i += 1
        line = line.rstrip()
        try:
            part_line = line.split("'vote_average': ")[1][:-1]
            [va, vc] = part_line.split(", 'vote_count': ")
            vote_count.append(int(vc))
            vote_average.append(float(va))
        except:
            vote_count.append(math.nan)
            vote_average.append(math.nan)
            print(line)
            print(i)
    votes = pd.DataFrame()
    votes['id'] = range(1, 7399)
    votes['vote_count'] = vote_count
    votes['vote_average'] = vote_average
    votes.to_csv('./input/votes.csv', index=False)


if __name__ == '__main__':
    rewrite()