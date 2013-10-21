# -*- coding: utf-8 -*-

from collections import defaultdict
import json
from crawler.play import output_path
import os
import codecs
from crawler.play.crawling import get_file_path, CommentExtractor
import re

__author__ = 'Luciana Dubiau'

def get_polarity(stars):
    if stars == 5:
        return 'pos'
    if stars == 1 or stars == 0:
        return 'neg'
    return None

def process(id_app):
    file_path = get_file_path(id_app)
    if os.path.isfile(file_path):
        with codecs.open(file_path, 'r', 'utf-8') as f:
            data = json.load(f)

        comments = defaultdict(lambda: [])
        for comment in data:
            stars = comment['stars']
            text = comment['text']
            title = comment['title']
            polarity = get_polarity(stars)
            full_text = title + ' ' + text
            words = full_text.split()
            if polarity and len(words) >= 5:
                comments[polarity].append(full_text)

        for polarity in comments:
            output_file_path = get_output_file_path(id_app, polarity)
            with codecs.open(output_file_path, 'w', 'utf-8') as f:
                json.dump(comments[polarity], f, indent=2)

def get_output_file_path(id_app, polarity):
    return os.path.join(get_output_path(polarity), '{}.json'.format(id_app))

def get_output_path(polarity):
    return os.path.join(os.path.join(output_path, 'output'), polarity)

def get_full_path():
    return os.path.join(os.path.join(output_path, 'full'))

def main():
    for polarity in ('pos', 'neg'):
        path = get_output_path(polarity)
        for file_name in os.listdir(path):
            f = os.path.join(path, file_name)
            if os.path.isfile(f):
                os.unlink(f)
    
    for file_name in filter(lambda x: x.endswith('json'), sorted(os.listdir(get_full_path()))):
        m = re.match(r'(.*)\.json', file_name)
        id = m.groups()[0]
        process(id)

if __name__ == '__main__':
    main()
