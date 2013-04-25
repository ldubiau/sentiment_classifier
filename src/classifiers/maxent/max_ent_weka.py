import nltk, os

from classifiers import base_path
from classifiers.classifier import SupervisedClassifier
from classifiers.evaluation import Evaluation
from classifiers import logger
from collections import defaultdict
from nltk.classify.weka import WekaClassifier
from nltk.classify import weka
from classifiers.extractors import BigramFeatureExtractor

weka_path = os.path.join(base_path, 'externaltools', 'weka.jar')

class CrossValidatedWekaMaxEntClassifier(SupervisedClassifier):

    def __init__(self, n_folds, fold_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro):
        super(CrossValidatedWekaMaxEntClassifier, self).__init__(n_folds, fold_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro)
        weka.config_weka(weka_path)
        weka.config_java(options='-Xmx1000M')


    def train(self, training_documents, feature_extractor):
        logger.info('Creating training dataset, documents size {}'.format(len(training_documents)))
        training_set = nltk.classify.util.apply_features(feature_extractor.extract, training_documents)
        logger.info('Training classifier')
        options = []
        options.append('-no-cv')
        self.classifier = nltk.WekaClassifier.train('weka.model', training_set, 'log_regression', options)
        
    def classify_comments(self, test_comments, feature_extractor):
        evaluation = Evaluation('pos', 'neg')
        test_set = nltk.classify.util.apply_features(feature_extractor.extract, test_comments)
        klasses = self.classifier.batch_classify(test_set)
        i = 0
        for test_comment, expected_klass in test_comments:
            evaluation.add(expected_klass, klasses[i])
            i += 1
        return evaluation
