# -*- coding: utf-8 -*
import nltk
from nltk import bigrams

__author__ = 'Luciana Dubiau'

class FeatureExtractor(object):

    def extract(self, document):
        pass

    def extract_features_names(self):
        pass

    def get_features_size(self):
        pass
        

class VocabularyFeatureExtractor(FeatureExtractor):

    def __init__(self, vocabulary):
        super(VocabularyFeatureExtractor, self).__init__()
        self.vocabulary = vocabulary

    def extract(self, document):
        features = {}
        for word in self.vocabulary:
            features.update(self.__process_word__(word, document))
        return features
        
    def extract_features_names(self):
        pass
        
    def __process_word__(self, word, document):
        return []

    def get_features_size(self):
        return len(self.vocabulary)

class WordPresenceFeatureExtractor(VocabularyFeatureExtractor):

    def __process_word__(self, word, document):
        return [('contains({})'.format(word.encode('utf-8')), word in document)]

    def extract_features_names(self):
        features_names = []
        for word in self.vocabulary:
            features_names.append(('contains({})'.format(word.encode('utf-8')), '{True, False}'))
        return features_names
        
    def __repr__(self):
        return 'word presence'


class WordFrequencyFeatureExtractor(VocabularyFeatureExtractor):

    def __process_word__(self, word, document):
        return [('count({})'.format(word.encode('utf-8')), len(filter(lambda x: x == word, document)))]

    def extract_features_names(self):
        features_names = []
        for word in self.vocabulary:
            features_names.append(('count({})'.format(word.encode('utf-8')), 'NUMERIC'))
        return features_names
        
    def __repr__(self):
        return 'word frequency'

class AllBigramsFeatureExtractor(FeatureExtractor):

    def __init__(self, bigrams):
        super(AllBigramsFeatureExtractor, self).__init__()
        self.bigrams = bigrams

    def extract(self, document):
        features = {}
        
        for bigram in self.bigrams:
            features['contains bigram({}, {})'.format(bigram[0].encode('utf-8'), bigram[1].encode('utf-8'))] = bigram in bigrams(document)
            
        return features

    def extract_features_names(self): 
        features_names = []
         
        for bigram in self.bigrams:
            features_names.append(('contains bigram({}, {})'.format(bigram[0].encode('utf-8'), bigram[1].encode('utf-8')), '{True, False}'))
           
        return features_names
    
    def get_features_size(self):
        return len(self.bigrams)

    def __repr__(self):
        return 'bigrams'
        
class BigramFeatureExtractor(FeatureExtractor):

    def __init__(self):
        super(BigramFeatureExtractor, self).__init__()

    def extract(self, document):
        features = {}
        
        for bigram in bigrams(document):
            features['contains bigram({}, {})'.format(bigram[0].encode('utf-8'), bigram[1].encode('utf-8'))] = True            
        return features

    def extract_features_names(self): 
        features_names = []
         
        for bigram in self.bigrams:
            features_names.append(('contains bigram({}, {})'.format(bigram[0].encode('utf-8'), bigram[1].encode('utf-8')),'{True, False}'))
           
        return features_names
        
    def __repr__(self):
        return 'bigrams'


class CompositeFeatureExtractor(FeatureExtractor):

    def __init__(self, extractors):
        super(CompositeFeatureExtractor, self).__init__()
        self.extractors = extractors

    def extract(self, document):
        features = {}
        for extractor in self.extractors:
            features.update(extractor.extract(document))
        return features

    def extract_features_names(self): 
        features_names = []
         
        for extractor in self.extractors:
            features_names.extend(extractor.extract_features_names())
           
        return set(features_names)
   
    def get_features_size(self): 
        size = 0
        for extractor in self.extractors:
            size += extractor.get_features_size()

        print size
        return size

    def __repr__(self):
        return 'composite extractor: {}'.format(', '.join([str(e) for e in self.extractors]))


def contains_sublist(lst, sublst):
    n = len(sublst)
    return any((sublst == lst[i:i + n]) for i in xrange(len(lst) - n + 1))
