# -*- coding: utf-8 -*-
import json
import codecs
import os
import nltk

from prettytable import PrettyTable
from classifiers import logger, base_path
from classifiers.evaluation import Evaluation
from classifiers.extractors import WordFrequencyFeatureExtractor, WordPresenceFeatureExtractor, BigramFeatureExtractor, AllBigramsFeatureExtractor, CompositeFeatureExtractor
from classifiers.processors import TokenizerProcessor, LowerCaseProcessor, TransformNegativeWordsProcessor, TransformCharactersProcessor, FilterPunctuationProcessor, FilterStopWordsProcessor, FilterWordLengthProcessor, TransformDuplicatedCharsProcessor, CompositeDocumentProcessor, StemmerProcessor
from collections import defaultdict
from nltk import bigrams

__author__ = 'Luciana Dubiau'

class Classifier(object):

    def __init__(self, n_folds, fold_size, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, min_word_length, remove_duplicated_chars, process_negation, stemming):
        self.pos_comments = []
        self.neg_comments = []
        self.n_folds = n_folds
        self.fold_size = fold_size
        self.remove_stop_words = remove_stop_words
        self.use_unigrams = use_unigrams
        self.use_unigrams_frequency = use_unigrams_frequency
        self.use_bigrams = use_bigrams
        self.min_word_length = min_word_length
        self.remove_duplicated_chars = remove_duplicated_chars
        self.process_negation = process_negation
        self.stemming = stemming

    def classify(self):
        self.load_corpus()
        self.preprocess_corpus()
        self.process_corpus()

    def load_corpus(self):
        self.pos_comments = self.load_comments(os.path.join(base_path, 'data', 'output', 'pos'))
        logger.info('Positive dataset loaded, size: {}'.format(len(self.pos_comments)))

        self.neg_comments = self.load_comments(os.path.join(base_path, 'data', 'output', 'neg'))
        logger.info('Negative dataset loaded, size: {}'.format(len(self.neg_comments)))

        if not self.fold_size:
            self.fold_size = min(len(self.pos_comments), len(self.neg_comments)) / self.n_folds

    def build_preprocessor(self):
        processors = [TokenizerProcessor(), LowerCaseProcessor(), TransformCharactersProcessor()]

        if self.remove_stop_words:
            processors.append(FilterStopWordsProcessor())

        if self.stemming:
            processors.append(StemmerProcessor())

        if self.min_word_length:
            processors.append(FilterWordLengthProcessor(self.min_word_length))

        if self.process_negation:
            processors.append(TransformNegativeWordsProcessor())

        processors.extend([FilterPunctuationProcessor()])

        if self.remove_duplicated_chars:
            processors.append(TransformDuplicatedCharsProcessor())

        return CompositeDocumentProcessor(processors)

    def preprocess_corpus(self):
        logger.info("Preprocessing Corpus")
        document_processor = self.build_preprocessor()
        self.pos_comments = [document_processor.process(c) for c in self.pos_comments[:self.fold_size * self.n_folds]]
        self.neg_comments = [document_processor.process(c) for c in self.neg_comments[:self.fold_size * self.n_folds]]

    def build_table(self):
        table = PrettyTable(['Fold', 'tp(pos)', 'cases(pos)', 'prec(pos)', 'recall(pos)', 'F1(pos)',
                             'tp(neg)', 'cases(neg)', 'prec(neg)', 'recall(neg)', 'F1(neg)', 'accuracy'])
        table.float_format = '1.3'
        for column_name in table.align:
            table.align[column_name] = 'r'
        table.align['Fold'] = 'c'
        return table

    def process_corpus(self):
        total_evaluation = Evaluation('pos', 'neg')
        table = self.build_table()

        vocabulary = set(sum([c for c in self.pos_comments + self.neg_comments], []))
        
        feature_extractors = []
        if self.use_unigrams:
            feature_extractors.append(WordPresenceFeatureExtractor(vocabulary))

        if self.use_unigrams_frequency:
            feature_extractors.append(WordFrequencyFeatureExtractor(vocabulary))

        if self.use_bigrams:
            feature_extractors.append(BigramFeatureExtractor())

        feature_extractor = CompositeFeatureExtractor(feature_extractors)
        
        for fold in range(self.n_folds):
            fold_start = fold*self.fold_size
            fold_end = fold_start + self.fold_size
            
            pos_fold_data = self.pos_comments[fold_start:fold_end]
            neg_fold_data = self.neg_comments[fold_start:fold_end]
            
            pos_labeled_comments = zip(pos_fold_data, ['pos'] * len(pos_fold_data))
            neg_labeled_comments = zip(neg_fold_data, ['neg'] * len(neg_fold_data))
            
            logger.info("Extracting Features Fold %s" %(fold+1))
            pos_featured_labeled_comments = nltk.classify.util.apply_features(feature_extractor.extract, pos_labeled_comments)
            neg_featured_labeled_comments = nltk.classify.util.apply_features(feature_extractor.extract, neg_labeled_comments)
                    
            logger.info("Building Frequency Distributions %s" %(fold+1))
            fold_featured_labeled_comments = pos_featured_labeled_comments + neg_featured_labeled_comments
            
            self.build_nltk_freq_distributions(fold, fold_featured_labeled_comments)
        
        for test_fold in range(self.n_folds):
            logger.info("Classifying Test Fold %s" %(test_fold+1))
            
            test_start = test_fold*self.fold_size
            test_end = test_start + self.fold_size
            
            pos_test_comments = self.pos_comments[test_start:test_end]
            neg_test_comments = self.neg_comments[test_start:test_end]
            
            pos_test_labeled_comments = zip(pos_test_comments, ['pos'] * len(pos_test_comments))
            neg_test_labeled_comments = zip(neg_test_comments, ['neg'] * len(neg_test_comments))
            
            pos_test_featured_labeled_comments = nltk.classify.util.apply_features(feature_extractor.extract, pos_test_labeled_comments)
            neg_test_featured_labeled_comments = nltk.classify.util.apply_features(feature_extractor.extract, neg_test_labeled_comments)
            test_comments = pos_test_featured_labeled_comments + neg_test_featured_labeled_comments
            
            evaluation = self.process_test_fold(test_fold, test_comments)
            total_evaluation.update(evaluation)

            table.add_row([test_fold,
                           evaluation.klasses['pos']['true_positives'],
                           evaluation.get_cases('pos'),
                           evaluation.get_precision('pos'),
                           evaluation.get_recall('pos'),
                           evaluation.get_f_measure('pos'),
                           evaluation.klasses['neg']['true_positives'],
                           evaluation.get_cases('neg'),
                           evaluation.get_precision('neg'),
                           evaluation.get_recall('neg'),
                           evaluation.get_f_measure('neg'),
                           evaluation.get_accuracy()])

        table.add_row(['Total',
                       total_evaluation.klasses['pos']['true_positives'],
                       total_evaluation.get_cases('pos'),
                       total_evaluation.get_precision('pos'),
                       total_evaluation.get_recall('pos'),
                       total_evaluation.get_f_measure('pos'),
                       total_evaluation.klasses['neg']['true_positives'],
                       total_evaluation.get_cases('neg'),
                       total_evaluation.get_precision('neg'),
                       total_evaluation.get_recall('neg'),
                       total_evaluation.get_f_measure('neg'),
                       total_evaluation.get_accuracy()])

        logger.info('Fold Size: {} - Avg Accuracy: {}'.format(self.fold_size, total_evaluation.get_accuracy()))
        print table

    def classify_comments(self, test_comments, feature_extractor):
        pass

    def load_comments(self, file_path):
        comments = []
        for file_name in filter(lambda x: x.endswith('json'), sorted(os.listdir(file_path))):
            with codecs.open(os.path.join(file_path, file_name), 'r', 'utf-8') as f:
                comments.extend(json.load(f))
        return comments

    def build_nltk_freq_distributions(self, fold, labeled_featuresets):
        pass 
        
    def process_test_fold(self, test_fold, test_comments):        
        evaluation = self.classify_comments(test_fold, test_comments)
        logger.info('TestSet Size: {} - Accuracy: {}'.format(evaluation.get_cases(), evaluation.get_accuracy()))
        return evaluation

