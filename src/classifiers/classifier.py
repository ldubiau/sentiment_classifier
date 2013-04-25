# -*- coding: utf-8 -*-
import json
import codecs
import os
import nltk
import pickle

from prettytable import PrettyTable
from classifiers import logger, base_path
from classifiers.evaluation import Evaluation
from classifiers.extractors import WordFrequencyFeatureExtractor, WordPresenceFeatureExtractor, BigramFeatureExtractor, AllBigramsFeatureExtractor, CompositeFeatureExtractor
from classifiers.processors import TokenizerProcessor, LowerCaseProcessor, TransformNegativeWordsProcessor, TransformCharactersProcessor, FilterPunctuationProcessor, FilterStopWordsProcessor, FilterWordLengthProcessor, TransformDuplicatedCharsProcessor, StemmerProcessor, CompositeCorpusProcessor, FreeLingProcessor
from collections import defaultdict
from nltk import bigrams
from nltk.probability import FreqDist

__author__ = 'Luciana Dubiau'

class Classifier(object):
    def __init__(self, remove_stop_words, min_word_length, remove_duplicated_chars, process_negation, stemming, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro):
        self.pos_comments = []
        self.neg_comments = []
        self.remove_stop_words = remove_stop_words
        self.min_word_length = min_word_length
        self.remove_duplicated_chars = remove_duplicated_chars
        self.process_negation = process_negation
        self.stemming = stemming
        self.transform_lower_case = transform_lower_case
        self.remove_punctuation_marks = remove_punctuation_marks
        self.remove_accents = remove_accents
        self.use_lemma = lemma
        self.adjectives = adjectives
        self.all_preprocessors = allprepro
        self.original_pos_comments = []
        self.original_neg_comments = []
        self.original_test_comments = []

        if self.all_preprocessors and (not os.path.isfile(os.path.join(base_path, 'data', 'pickle', 'pos_comments.pickle')) or not os.path.isfile(os.path.join(base_path, 'data', 'pickle', 'neg_comments.pickle'))):
            self.dump_preprocessed_corpus()

    def classify(self):
        if self.all_preprocessors:
            self.load_preprocessed_corpus()
        else:
            self.load_corpus()
            self.preprocess_corpus()
        return self.process_corpus()

    def load_comments(self, file_path):
        comments = []
        for file_name in filter(lambda x: x.endswith('json'), sorted(os.listdir(file_path))):
            with codecs.open(os.path.join(file_path, file_name), 'r', 'utf-8') as f:
                comments.extend(json.load(f))
        return comments

    def load_corpus(self):
        self.pos_comments = self.load_comments(os.path.join(base_path, 'data', 'output', 'pos'))
        logger.info('Positive dataset loaded, size: {}'.format(len(self.pos_comments)))

        self.neg_comments = self.load_comments(os.path.join(base_path, 'data', 'output', 'neg'))
        logger.info('Negative dataset loaded, size: {}'.format(len(self.neg_comments)))

        self.original_pos_comments = self.pos_comments
        self.original_neg_comments = self.neg_comments

    def build_preprocessor(self):
        processors = [TokenizerProcessor()]

        if self.transform_lower_case:
            processors.append(LowerCaseProcessor())

        if self.remove_duplicated_chars:
            processors.append(TransformDuplicatedCharsProcessor())

        if self.remove_accents or self.all_preprocessors:
            processors.append(TransformCharactersProcessor())

        if self.use_lemma:
            processors.append(FreeLingProcessor())

        if self.adjectives:
            processors.append(FreeLingProcessor(lambda term: term.tag.startswith('AQ')))

        if self.remove_stop_words or self.all_preprocessors:
            processors.append(FilterStopWordsProcessor())

        if self.stemming:
            processors.append(StemmerProcessor())

        if self.min_word_length or self.all_preprocessors:
            if not self.min_word_length:
                self.min_word_length = 3
            processors.append(FilterWordLengthProcessor(self.min_word_length))

        if self.process_negation or self.all_preprocessors:
            processors.append(TransformNegativeWordsProcessor())

        if self.remove_punctuation_marks or self.all_preprocessors:
            processors.append(FilterPunctuationProcessor())

        return CompositeCorpusProcessor(processors)

    def preprocess_corpus(self):
        logger.info("Preprocessing corpus")
   
        processor = self.build_preprocessor()
        self.pos_comments = processor.process(self.pos_comments)
        self.neg_comments = processor.process(self.neg_comments)


    def dump_preprocessed_corpus(self):
        logger.info("Loading corpus")
        self.load_corpus()

        logger.info("Preprocessing corpus")
        document_processor = self.build_preprocessor()
        self.pos_comments = [document_processor.process(c) for c in self.pos_comments]
        self.neg_comments = [document_processor.process(c) for c in self.neg_comments]

        logger.info("Creating pickle files")
        pickle.dump(self.pos_comments, open(os.path.join(base_path, 'data', 'pickle', 'pos_comments.pickle'), 'wb'))
        pickle.dump(self.neg_comments, open(os.path.join(base_path, 'data', 'pickle', 'neg_comments.pickle'), 'wb'))

    def load_preprocessed_corpus(self):
        logger.info("Loading pickle files")
        self.pos_comments = pickle.load(open(os.path.join(base_path, 'data', 'pickle', 'pos_comments.pickle'), 'rb'))
        self.neg_comments = pickle.load(open(os.path.join(base_path, 'data', 'pickle', 'neg_comments.pickle'), 'rb'))


    def process_corpus(self):
        pass

 
class SupervisedClassifier(Classifier):

    def __init__(self, n_folds, fold_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stemming, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro):
        super(SupervisedClassifier, self).__init__(remove_stop_words, min_word_length, remove_duplicated_chars, process_negation, stemming, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro)
        self.n_folds = n_folds
        self.fold_size = fold_size
        self.fold_number = fold_number
        self.use_unigrams = use_unigrams
        self.use_unigrams_frequency = use_unigrams_frequency
        self.use_bigrams = use_bigrams
        self.use_all_bigrams = use_all_bigrams

    def load_corpus(self):
        super(SupervisedClassifier, self).load_corpus()

        if not self.fold_size:
            self.fold_size = min(len(self.pos_comments), len(self.neg_comments)) / self.n_folds

    def preprocess_corpus(self):
        logger.info("Preprocessing corpus")
  
        processor = self.build_preprocessor()
        self.pos_comments = processor.process(self.pos_comments[:self.fold_size * self.n_folds])
        self.neg_comments = processor.process(self.neg_comments[:self.fold_size * self.n_folds])

    def load_preprocessed_corpus(self):
        logger.info("Loading pickle files")
        self.pos_comments = pickle.load(open(os.path.join(base_path, 'data', 'pickle', 'pos_comments.pickle'), 'rb'))[:self.fold_size * self.n_folds]
        self.neg_comments = pickle.load(open(os.path.join(base_path, 'data', 'pickle', 'neg_comments.pickle'), 'rb'))[:self.fold_size * self.n_folds]
        
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
        
        folds = []
        
        if self.fold_number is None:
            folds = range(self.n_folds)
        else:
            folds.append(self.fold_number)

        for i in folds:
            training_documents, pos_test_set, neg_test_set = self.get_data_sets(i)

            feature_extractors = []
            if self.use_unigrams:
                bag_of_words = self.get_bag_of_words(training_documents)
                feature_extractors.append(WordPresenceFeatureExtractor(bag_of_words))

            if self.use_unigrams_frequency:
                bag_of_words = self.get_bag_of_words(training_documents)
                feature_extractors.append(WordFrequencyFeatureExtractor(bag_of_words))

            if self.use_bigrams:
                feature_extractors.append(BigramFeatureExtractor())
                
            if self.use_all_bigrams:
                top_bigrams = self.get_top_bigrams(training_documents)
                feature_extractors.append(AllBigramsFeatureExtractor(top_bigrams))

            feature_extractor = CompositeFeatureExtractor(feature_extractors)

            evaluation = self.process_fold(training_documents, pos_test_set, neg_test_set, feature_extractor)
            total_evaluation.update(evaluation)

            table.add_row([i,
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

        logger.info('Total TestSet Size: {} - Avg Accuracy: {}'.format(total_evaluation.get_cases(), total_evaluation.get_accuracy()))
        print table
        return total_evaluation

    def classify_comments(self, test_comments, feature_extractor):
        evaluation = Evaluation('pos', 'neg')
        i = 0
        for comment, expected_klass in test_comments:
            klass = self.classifier.classify(feature_extractor.extract(comment))
            #if klass != expected_klass:
            #print 'expected class: %s, class: %s, comment: %s' %(expected_klass, klass, self.original_test_comments[i].encode('utf-8'))
            evaluation.add(expected_klass, klass)
            i = i + 1
        return evaluation

    def train(self, training_documents, feature_extractor):
        pass
        
    def process_fold(self, training_documents, pos_test_comments, neg_test_comments, feature_extractor):
        logger.info('Feature extractor: {}'.format(str(feature_extractor)))
        self.train(training_documents, feature_extractor)

        logger.info('Classifying')

        test_comments = zip(pos_test_comments, ['pos'] * len(pos_test_comments))
        test_comments.extend(zip(neg_test_comments, ['neg'] * len(neg_test_comments)))

        evaluation = self.classify_comments(test_comments, feature_extractor)

        logger.info('TestSet Size: {} - Accuracy: {}'.format(evaluation.get_cases(), evaluation.get_accuracy()))

        return evaluation

    def get_bag_of_words(self, training_documents):
        bag_of_words_freq = FreqDist()
        for w in sum([d[0] for d in training_documents], []):
            bag_of_words_freq.inc(w)

        min_freq = 10
        if self.adjectives:
            min_freq = 4

        bag_of_words = filter(lambda x: bag_of_words_freq[x] > min_freq, bag_of_words_freq.keys())
        
        bag_of_words = bag_of_words[:3000]
        logger.info('bag of words size: {}'.format(len(bag_of_words)))

        return bag_of_words

    def get_top_bigrams(self, training_documents):
        all_bigrams_freq = FreqDist()
        all_bigrams = []

        for bi in sum([bigrams(d[0]) for d in training_documents], []):
            all_bigrams_freq.inc(bi)

        top_bigrams = filter(lambda x: all_bigrams_freq[x] > 4, all_bigrams_freq.keys())

        top_bigrams = top_bigrams[:3000]
        print "bigrams size: " + str(len(top_bigrams))

        return top_bigrams
        
    def get_data_sets(self, iteration):
        test_start = self.fold_size * iteration
        test_end = test_start + self.fold_size

        logger.info('Fold {}/{}, test set {}..{}'.format(iteration + 1, self.n_folds, test_start, test_end))

        training_documents = [(c, 'pos') for c in self.pos_comments[:test_start] + self.pos_comments[test_end:]]
        training_documents.extend([(c, 'neg') for c in self.neg_comments[:test_start] + self.neg_comments[test_end:]])

        pos_test_set = self.pos_comments[test_start:test_end]
        neg_test_set = self.neg_comments[test_start:test_end]

        self.original_test_comments = self.original_pos_comments[test_start:test_end] + self.original_neg_comments[test_start:test_end]

        return training_documents, pos_test_set, neg_test_set


