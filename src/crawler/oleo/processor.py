# -*- coding: utf-8 -*-

from collections import defaultdict
import json
from crawler.oleo import LAST_RESTO_ID, output_path
import os
import codecs
from crawler.oleo.crawling import get_file_path, CommentExtractor

__author__ = 'Luciana Dubiau'

FILTERS_BY_POLARITY = {'pos': CommentExtractor.FILTERS[0:2], 'neg': CommentExtractor.FILTERS[2:]}

def get_polarity_by_classification(comment, classification):
    return 'pos' if classification in CommentExtractor.FILTERS[0:2] else 'neg'

def get_polarity_by_ranking(comment, classification):
    total = sum([comment['ranking'][x] for x in comment['ranking']])
    comida = comment['ranking']['Comida']
    servicio = comment['ranking']['Servicio']
    ambiente = comment['ranking']['Ambiente']
    if total > 10:
        return 'pos'
    if (comida == 1) or (comida == 2 and total == 4):
        return 'neg'
    return None

get_polarity = get_polarity_by_ranking

def process(id_resto):
    file_path = get_file_path(id_resto)
    if os.path.isfile(file_path):
        with codecs.open(file_path, 'r', 'utf-8') as f:
            data = json.load(f)
        comments = defaultdict(lambda: [])
        for classification in data:
            for comment in data[classification]:
                polarity = get_polarity(comment, classification)
                if polarity:
                    comments[polarity].append(comment['text'])
        for polarity in comments:
            output_file_path = get_output_file_path(id_resto, polarity)
            with codecs.open(output_file_path, 'w', 'utf-8') as f:
                json.dump(comments[polarity], f, indent=2)

def get_output_file_path(id_resto, polarity):
    return os.path.join(get_output_path(polarity), '{}.json'.format(id_resto))

def get_output_path(polarity):
    return os.path.join(os.path.join(output_path, 'output'), polarity)

def main():
    for polarity in ('pos', 'neg'):
        path = get_output_path(polarity)
        for file_name in os.listdir(path):
            f = os.path.join(path, file_name)
            if os.path.isfile(f):
                os.unlink(f)
    for id in range(0, LAST_RESTO_ID):
        process(id)

if __name__ == '__main__':
    main()
