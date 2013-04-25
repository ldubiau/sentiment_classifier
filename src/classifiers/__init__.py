# -*- coding: utf-8 -*-
import logging
import sys
import nltk
import os

__author__ = 'Luciana Dubiau'

logger = logging.getLogger(__package__)
handler = logging.FileHandler('sentiment_classifier.log')
#handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

nltk_path = os.path.join(base_path, 'data', 'nltk')
nltk.data.path.append(nltk_path)
try:
    nltk.corpus.stopwords.words('spanish')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_path)

try:
    nltk.corpus.cess_esp.tagged_words()
except LookupError:
    nltk.download('cess_esp', download_dir=nltk_path)
