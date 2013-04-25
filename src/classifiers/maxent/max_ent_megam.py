# -*- coding: utf-8 -*
import nltk, os

from classifiers import base_path
from classifiers.classifier import SupervisedClassifier
from classifiers.evaluation import Evaluation
from classifiers import logger
from collections import defaultdict
from nltk.classify.maxent import MaxentClassifier
from nltk.classify import megam

megam_path = os.path.join(base_path, 'externaltools')

class CrossValidatedMegamMaxEntClassifier(SupervisedClassifier):
    
    def __init__(self, n_folds, fold_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro):
        super(CrossValidatedMegamMaxEntClassifier, self).__init__(n_folds, fold_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro)
        
        megam.config_megam(megam_path)
        
    def train(self, training_documents, feature_extractor):
        logger.info('Creating training dataset, documents size {}'.format(len(training_documents)))
        training_set = nltk.classify.util.apply_features(feature_extractor.extract, training_documents)
        logger.info('Training classifier')
        #self.classifier = nltk.MaxentClassifier.train(training_set, algorithm='megam', explicit=True, bernoulli=True, model='multiclass')
        self.classifier = nltk.MaxentClassifier.train(training_set, algorithm='megam')
        self.classifier.show_most_informative_features()
        
