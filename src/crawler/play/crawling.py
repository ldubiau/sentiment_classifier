# -*- coding: utf-8 -*-

import json
import httplib
import re
import httplib2
import codecs
import os
import json
import bs4

from crawler.play import logger, output_path, app_ids

__author__ = 'Luciana Dubiau'


class CommentExtractor(object):
    COMMENT_URL_TEMPLATE = 'https://play.google.com/store/getreviews?reviewType=0&pageNum={}&id={}&xhr=1'
    COMMENT_REGEX = '<div [^<>]*> <span [^<>]*>([^<>]*)</span>([^<>]*)<div [^<>]*> <a [^<>]*>[^<>]*</a> </div> </div>'
    RATING_REGEX = '<div class="current-rating" style="width: ([0-9]*).0%;"></div>'
    STARS_PATTERN = {
        "0": 0,
        "20": 1,
        "40": 2,
        "60": 3,
        "80": 4,
        "100": 5
    }

    def get_all_comments(self, id_app):
        page = 1
        more_comments = True
        last_result = ''
        data = []
        while more_comments:            
            http = httplib2.Http()
            url = CommentExtractor.COMMENT_URL_TEMPLATE.format(page, id_app)
            res, content = http.request(uri=url, method='POST', headers={'Content-length': '0', 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:22.0) Gecko/20100101 Firefox/22.0'})
            logger.info('Result {} from {}'.format(res.status, url))
            
            if res.status == httplib.OK:
                content = content[20:len(content)-5]
                content = content.replace('\\"', '"')
                content = content.replace('\u003c', '<')
                content = content.replace('\u003e', '>')
                content = content.replace('\u003d', '=')
                
                html = bs4.BeautifulSoup(content)
                
                if content == last_result: 
                    more_comments = False
                else:
                    last_result = content
                    comments = html.find_all('div', attrs={'class': 'single-review'})
                    for comment in comments:
                        body = comment.find_all('div', attrs={'class': 'review-body'})[0]

                        m = re.match(CommentExtractor.COMMENT_REGEX, str(body)) 
                        title = m.groups()[0]
                        text = m.groups()[1]
                        
                        rating = comment.find_all('div', attrs={'class': 'current-rating'})[0]
                        m = re.match(CommentExtractor.RATING_REGEX, str(rating)) 
                        stars = m.groups()[0]
                       
                        n_stars = CommentExtractor.STARS_PATTERN[stars]                        
                        
                        data.append({'title': title, 'text': text, 'stars': n_stars})
                    
                page = page + 1
            else:
                print res
                raise CrawlerError('Error {} accediendo la URL {}'.format(res.status, url))

            self.dump_data(id_app, data)

    def dump_data(self, id_app, data):
        file_path = get_file_path(id_app)
        with codecs.open(file_path, 'w', 'utf-8') as f:
            json.dump(data, f, indent=2)


def get_file_path(id_app):
    return os.path.join(os.path.join(output_path, 'full'), '{}.json'.format(id_app))


class CrawlerError(Exception):
    def __init__(self, message=None):
        super(CrawlerError, self).__init__()
        self.message = message


def main():
    extractor = CommentExtractor()
    
    for id in app_ids:
        extractor.get_all_comments(id)


if __name__ == '__main__':
    main()
