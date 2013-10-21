# -*- coding: utf-8 -*
import nltk, numpy
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from classifiers.classifier import SupervisedClassifier
from classifiers import logger

class CrossValidatedSciKitClassifier(SupervisedClassifier):
    
    def __init__(self, n_folds, corpus_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, out_of_domain_test, proportion_of_positives, algorithm):
        super(CrossValidatedSciKitClassifier, self).__init__(n_folds, corpus_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, out_of_domain_test, proportion_of_positives)
        self.algorithm = algorithm
        
    def train(self, training_documents, feature_extractor):
        logger.info('Creating training dataset, documents size {}'.format(len(training_documents)))
        #training_set = nltk.classify.util.apply_features(feature_extractor.extract, training_documents)

        training_set = []
        for td in training_documents:
            document = td[0]
            label = td[1]
            features = feature_extractor.extract(document)
            training_set.append((features, label))
                
        logger.info('Building classifier')
        if self.algorithm == 'nb':
            self.classifier = SklearnClassifier(MultinomialNB(), dtype=bool)
        elif self.algorithm == 'maxent':
            self.classifier = SklearnClassifier(LogisticRegression(), dtype=numpy.float64)
        elif self.algorithm == 'svm':
            self.classifier = SklearnClassifier(LinearSVC())
        elif self.algorithm == 'tree':
            self.classifier = SklearnClassifier(DecisionTreeClassifier(), sparse=False) #optimized version of the CART algorithm
            #dot_data = StringIO.StringIO() 
            #tree.export_graphviz(self.classifier._clf, dot_data, feature_names=self.classifier._feature_index.keys())        
            #graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
            #graph.write_pdf("test_export_graphvix.pdf")
        
        logger.info('Training classifier')
        self.classifier.train(training_set)        
        
