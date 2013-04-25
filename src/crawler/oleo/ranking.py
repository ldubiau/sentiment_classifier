# -*- coding: utf-8 -*-

import json
import codecs
from prettytable import PrettyTable
from crawler.oleo import LAST_RESTO_ID
import os
from crawler.oleo.crawling import get_file_path, CommentExtractor

__author__ = 'Luciana Dubiau'

def load(id_resto):
    file_path = get_file_path(id_resto)
    if os.path.isfile(file_path):
        with codecs.open(file_path, 'r', 'utf-8') as f:
            data = json.load(f)
        return data

def main():
    full_data = {}
    for qry_filter in CommentExtractor.FILTERS:
        full_data[qry_filter] = []
    for id in range(0, LAST_RESTO_ID):
        data = load(id)
        if data:
            for qry_filter in CommentExtractor.FILTERS:
                if qry_filter in data:
                    full_data[qry_filter].extend([comment['ranking'] for comment in data[qry_filter]])
    #features = set(sum([x.keys() for x in sum([full_data[x] for x in full_data], [])], []))
    features = ('Ambiente', 'Comida', 'Servicio')
    table = PrettyTable(['Filter', 'Feature', 'min', 'max'])
    table.align['Filter'] = 'l'
    table.align['Feature'] = 'l'
    table.align['min'] = 'r'
    table.align['max'] = 'r'
    for qry_filter in CommentExtractor.FILTERS:
        for feature in features:
            table.add_row([qry_filter, feature, min([x[feature] for x in full_data[qry_filter]]), max([x[feature] for x in full_data[qry_filter]])])
    print table
    print len([x for x in full_data['excelentes'] if len(list(y for y in x if x[y] == 4)) == 3])
    print len([x for x in full_data['malos'] if len(list(y for y in x if x[y] == 1)) == 3])

if __name__ == '__main__':
    main()
