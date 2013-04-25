# -*- coding: utf-8 -*
from classifiers.extractors import WordPresenceFeatureExtractor

import pytest

__author__ = 'Luciana Dubiau'

data = [
    (['hola', 'mundo', 'chau'], ['hola', 'mundo'])
]

@pytest.mark.parametrize(('vocabulary', 'document'), data)
def test_presence_extractor(vocabulary, document):
    extractor = WordPresenceFeatureExtractor(vocabulary)
    features = extractor.extract(document)
    for word in document:
        assert features['contains({})'.format(word)]
    for word in [w for w in vocabulary if w not in document]:
        assert not features['contains({})'.format(word)]
