# -*- coding: utf-8 -*-
import json
import codecs
import os
import nltk
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
    def __init__(self, corpus_size, remove_stop_words, min_word_length, remove_duplicated_chars, process_negation, stemming, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, out_of_domain_test, proportion_of_positives):
        self.pos_comments = []
        self.neg_comments = []
        self.pos_comments_dom2 = []
        self.neg_comments_dom2 = []
        self.corpus_size = corpus_size
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
        self.out_of_domain_test = out_of_domain_test
        self.prop_of_pos = proportion_of_positives
        self.prop_of_neg = 1 - proportion_of_positives

        self.original_pos_comments = []
        self.original_neg_comments = []
        self.original_test_comments = []

    def classify(self):
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

        if self.out_of_domain_test:
            self.pos_comments_dom2 = self.load_comments(os.path.join(base_path, 'data2', 'output', 'pos')) 
            logger.info('Positive dataset 2 loaded, size: {}'.format(len(self.pos_comments_dom2)))

            self.neg_comments_dom2 = self.load_comments(os.path.join(base_path, 'data2', 'output', 'neg'))
            logger.info('Negative dataset 2 loaded, size: {}'.format(len(self.neg_comments_dom2)))


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

        self.original_pos_comments = self.pos_comments
        self.original_neg_comments = self.neg_comments

        self.pos_comments = processor.process(self.pos_comments[:int(self.corpus_size*self.prop_of_pos)])
        self.neg_comments = processor.process(self.neg_comments[:int(self.corpus_size*self.prop_of_neg)])

        self.pos_comments_dom2 = processor.process(self.pos_comments_dom2[:int(self.corpus_size*self.prop_of_pos)])
        self.neg_comments_dom2 = processor.process(self.neg_comments_dom2[:int(self.corpus_size*self.prop_of_neg)])

    def process_corpus(self):
        pass

    def build_metrics_table(self):
        table = PrettyTable(['Fold', 'tp(p)', 'cases(p)', 'prec(p)', 'rec(p)', 'acc(p)', 'f1(p)',
                             'tp(n)', 'cases(n)', 'prec(n)', 'rec(n)', 'acc(n)', 'f1(n)', 'acc(avg)', 'f1(avg)'])
        table.float_format = '1.3'
        for column_name in table.align:
            table.align[column_name] = 'r'
        table.align['Fold'] = 'c'
        return table

    def add_metrics(self, metrics_table, fold, evaluation):
        metrics_table.add_row([fold,
                       evaluation.klasses['pos']['true_positives'],
                       evaluation.get_cases('pos'),
                       evaluation.get_precision('pos'),
                       evaluation.get_recall('pos'),
                       evaluation.get_accuracy('pos'),
                       evaluation.get_f_measure('pos'),
                       evaluation.klasses['neg']['true_positives'],
                       evaluation.get_cases('neg'),
                       evaluation.get_precision('neg'),
                       evaluation.get_recall('neg'),
                       evaluation.get_accuracy('neg'),
                       evaluation.get_f_measure('neg'),
                       evaluation.get_accuracy_avg(),
                       evaluation.get_f_measure_avg()])
 

class SupervisedClassifier(Classifier):

    def __init__(self, n_folds, corpus_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stemming, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, out_of_domain_test, proportion_of_positives):
        super(SupervisedClassifier, self).__init__(corpus_size, remove_stop_words, min_word_length, remove_duplicated_chars, process_negation, stemming, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, out_of_domain_test, proportion_of_positives)
        self.n_folds = n_folds

        self.cross_validation = self.n_folds > 0
        self.fold_number = fold_number
        self.use_unigrams = use_unigrams
        self.use_unigrams_frequency = use_unigrams_frequency
        self.use_bigrams = use_bigrams
        self.use_all_bigrams = use_all_bigrams

    def build_feature_extractor(self, training_documents):
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

        return feature_extractor

    def process_corpus_with_cross_validation(self, total_evaluation, metrics_table):
        folds = []

        if self.fold_number is None:
            folds = range(self.n_folds)
        else:
            folds.append(self.fold_number)

        pos_fold_size = int(self.corpus_size*self.prop_of_pos / self.n_folds)
        neg_fold_size = int(self.corpus_size*self.prop_of_neg / self.n_folds)
        for i in folds:
            pos_test_start = pos_fold_size * i
            pos_test_end = pos_test_start + pos_fold_size

            neg_test_start = neg_fold_size * i
            neg_test_end = neg_test_start + neg_fold_size

            logger.info('Fold {}/{}, pos: {}..{}, neg: {}..{}'.format(i + 1, self.n_folds, pos_test_start, pos_test_end, neg_test_start, neg_test_end))

            training_documents = [(c, 'pos') for c in self.pos_comments[:pos_test_start] + self.pos_comments[pos_test_end:]]
            training_documents.extend([(c, 'neg') for c in self.neg_comments[:neg_test_start] + self.neg_comments[neg_test_end:]])

            if self.out_of_domain_test:
                # testeo siempre con el mismo set en todas las iteraciones
                pos_test_set = self.pos_comments_dom2[:pos_fold_size]
                neg_test_set = self.neg_comments_dom2[:neg_fold_size]

                self.original_test_comments = self.original_pos_comments[:pos_fold_size]
                self.original_test_comments.extend(self.original_neg_comments[:neg_fold_size])

            else:
                pos_test_set = self.pos_comments[pos_test_start:pos_test_end]
                neg_test_set = self.neg_comments[neg_test_start:neg_test_end]

                self.original_test_comments = self.original_pos_comments[pos_test_start:pos_test_end]
                self.original_test_comments.extend(self.original_neg_comments[neg_test_start:neg_test_end])


            evaluation = self.process_fold(training_documents, pos_test_set, neg_test_set)
            total_evaluation.update(evaluation)

            self.add_metrics(metrics_table, i, evaluation)

    def process_corpus_with_holdout_validation(self, total_evaluation):
        pos_training_size = int(self.corpus_size*self.prop_of_pos / 3 * 2)
        neg_training_size = int(self.corpus_size*self.prop_of_neg / 3 * 2)

        training_documents = [(c, 'pos') for c in self.pos_comments[0:pos_training_size]]
        training_documents.extend([(c, 'neg') for c in self.neg_comments[0:neg_training_size]])

        if self.out_of_domain_test:
            pos_test_size = int(self.corpus_size*self.prop_of_pos / 3)
            neg_test_size = int(self.corpus_size*self.prop_of_neg / 3)

            pos_test_set = self.pos_comments_dom2[:pos_test_size]
            neg_test_set = self.neg_comments_dom2[:neg_test_size]
        else:
            pos_test_set = self.pos_comments[pos_training_size:]
            neg_test_set = self.neg_comments[neg_training_size:]

        evaluation = self.process_fold(training_documents, pos_test_set, neg_test_set)
        total_evaluation.update(evaluation)

    def process_corpus(self):
        total_evaluation = Evaluation('pos', 'neg')
        metrics_table = self.build_metrics_table()
       
        if (self.cross_validation): 
            self.process_corpus_with_cross_validation(total_evaluation, metrics_table)
        else:
            self.process_corpus_with_holdout_validation(total_evaluation)

        self.add_metrics(metrics_table, 'Total', total_evaluation)

        logger.info('Total TestSet Size: {} - Avg Accuracy: {}'.format(total_evaluation.get_cases(), total_evaluation.get_accuracy_avg()))
        print metrics_table
        return total_evaluation

    def classify_comments(self, test_comments, feature_extractor):
        evaluation = Evaluation('pos', 'neg')
        i = 0
        for comment, expected_klass in test_comments:
            klass = self.classifier.classify(feature_extractor.extract(comment))
            #if klass != expected_klass:
            print "class: " + klass + " - expected class: " + expected_klass + ": " + self.original_test_comments[i].encode('utf-8')
            evaluation.add(expected_klass, klass)
            i = i + 1
        return evaluation

    def train(self, training_documents, feature_extractor):
        pass
        
    def process_fold(self, training_documents, pos_test_comments, neg_test_comments):
        feature_extractor = self.build_feature_extractor(training_documents)

        logger.info('Feature extractor: {}'.format(str(feature_extractor)))
        self.train(training_documents, feature_extractor)

        logger.info('Classifying')

        test_comments = zip(pos_test_comments, ['pos'] * len(pos_test_comments))
        test_comments.extend(zip(neg_test_comments, ['neg'] * len(neg_test_comments)))

        evaluation = self.classify_comments(test_comments, feature_extractor)

        logger.info('TestSet Size: {} - Accuracy: {}'.format(evaluation.get_cases(), evaluation.get_accuracy_avg()))

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
        

