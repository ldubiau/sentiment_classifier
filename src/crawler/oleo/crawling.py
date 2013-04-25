# -*- coding: utf-8 -*-

import json
import httplib
import re
import httplib2
import bs4
import codecs
import os

from crawler.oleo import logger, LAST_RESTO_ID, output_path

__author__ = 'Luciana Dubiau'

class CommentExtractor(object):

    FILTERS = ('excelentes', 'buenos', 'malos')
    DEFAULT_FILTER = FILTERS[0]
    DEFAULT_ORDER = ''
    COMMENT_URL_TEMPLATE = 'http://www.guiaoleo.com.ar/restaurante/getcomentarios/idResto/{}/filterQry/{}/orderQry/{}/offset/{}'
    PAGE_LENGTH = 5

    def get_all_comments(self, id_resto, qry_order=DEFAULT_ORDER):
        data = {}
        for idx, qry_filter in enumerate(CommentExtractor.FILTERS):
            comments, totals = self.get_all_comments_by_filter(id_resto, qry_filter, qry_order)
            if sum(totals) == 0:
                return
            assert len(comments) == totals[idx + 1], 'Expected {} comments, got {}.'.format(totals[idx + 1], len(comments))
            if len(comments) > 0:
                data[CommentExtractor.FILTERS[idx]] = comments
        if len(data) > 0:
            file_path = get_file_path(id_resto)
            with codecs.open(file_path, 'w', 'utf-8') as f:
                json.dump(data, f, indent=2)

    def get_all_comments_by_filter(self, id_resto, qry_filter=DEFAULT_FILTER, qry_order=DEFAULT_ORDER):
        all_comments = []
        offset = 0
        condition = True
        while condition:
            comments, totals = self.get_comments(id_resto, offset, qry_filter, qry_order)
            all_comments.extend(comments)
            condition = len(comments) >= CommentExtractor.PAGE_LENGTH
            offset += CommentExtractor.PAGE_LENGTH
        return all_comments, totals

    def get_comments(self, id_resto, qry_offset=0, qry_filter=DEFAULT_FILTER, qry_order=DEFAULT_ORDER):
        http = httplib2.Http()
        url = CommentExtractor.COMMENT_URL_TEMPLATE.format(id_resto, qry_filter, qry_order, qry_offset)
        res, content = http.request(url)
        logger.info('Result {} from {}'.format(res.status, url))
        if res.status == httplib.OK:
            html = bs4.BeautifulSoup(content)
            totals = self.get_totals(html)
            logger.info('Totals for {} - excelentes {} - buenos {} - malos {}'.format(url, totals[1], totals[2], totals[3]))
            comments_text = []
            comments = html.find_all('div', attrs={'class': 'userStory comment'})
            for comment in comments:
                href = comment.find('a', attrs={'href': lambda x: x.startswith('/members/'), 'class': None}).attrs['href']
                user_name = href[len('/members/'):]
                date = comment.find('p', attrs={'class': 'userStoryText'}).find('span', attrs={'class': 'commentDate'}).text
                ranking = self.get_ranking(comment)
                p = comment.find('p', {'class': 'userStoryText'})
                text = unicode('\n'.join([s.strip() for s in p.contents if isinstance(s, bs4.element.NavigableString) and not isinstance(s, bs4.element.Comment) and len(s.strip()) > 0]))
                comments_text.append({'user_name': user_name, 'date': date, 'ranking': ranking, 'text': text})
            return comments_text, totals
        else:
            raise CrawlerError('Error {} accediendo la URL {}'.format(res.status, url))

    def get_totals(self, html):
        totals = [0, 0, 0, 0]
        for content in [s.replace('\n', '').strip() for s in html.contents if isinstance(s, bs4.element.NavigableString)]:
            if totals[0] == 0:
                match = re.match('(\d+)\|#\|', content)
                if match:
                    totals[0] = int(match.groups()[0])
            match = re.match('.*\|#\|(\d+)\|#\|(\d+)\|#\|(\d+)', content)
            if match:
                totals[1:4] = [int(n) for n in match.groups()]
        return tuple(totals)

    def get_ranking(self, comment):
        ranking = {}
        ul = comment.find('ul', attrs={'class': 'restoDatosRanking collection'})
        if ul:
            categories = ul.find_all('li')
            for category in categories:
                name = category.find('label').text[:-1]
                points = len(category.find_all('span', attrs={'class': 'rankBox on'}))
                ranking[name] = points
        return ranking

def get_file_path(id_resto):
    return os.path.join(os.path.join(output_path, 'full'), '{}.json'.format(id_resto))

class CrawlerError(Exception):

    def __init__(self, message=None):
        super(CrawlerError, self).__init__()
        self.message = message

def main():
    extractor = CommentExtractor()
    #for id in range(12599, LAST_RESTO_ID):
    for id in range(18000, LAST_RESTO_ID):
        extractor.get_all_comments(id)

if __name__ == '__main__':
    main()
