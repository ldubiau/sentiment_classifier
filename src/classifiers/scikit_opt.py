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
    
    def __init__(self, n_folds, fold_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, algorithm):
        super(CrossValidatedSciKitClassifier, self).__init__(n_folds, fold_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro)
        self.algorithm = algorithm
        
    def train(self, training_documents, feature_extractor):
        logger.info('Creating training dataset, documents size {}'.format(len(training_documents)))
        #training_set = nltk.classify.util.apply_features(feature_extractor.extract, training_documents)

        featuresets = []
        labelsets = []
        feature_index = {}
        labels = ['pos', 'neg']
        label_index = {'neg': 1, 'pos': 0}
        X = numpy.zeros((len(training_documents), feature_extractor.get_features_size()), dtype=bool)
        i = 0
        for td in training_documents:
            #print "Extracting Features: Document " + str(i)
            document = td[0]
            label = td[1]
            features = feature_extractor.extract(document)
            featuresets.append(features)
            labelsets.append(label)

            for f,v in features.iteritems():
                if f not in feature_index:
                    feature_index[f] = len(feature_index)

                X[i, feature_index[f]] = bool(v)

            i = i + 1
                
        logger.info('Building classifier')
        if self.algorithm == 'nb':
            self.classifier = SklearnClassifier(MultinomialNB(), dtype=bool)
        elif self.algorithm == 'maxent':
            self.classifier = SklearnClassifier(LogisticRegression(), dtype=numpy.float64)
        elif self.algorithm == 'svm':
            self.classifier = SklearnClassifier(LinearSVC(), sparse=False)
        elif self.algorithm == 'tree':
            self.classifier = SklearnClassifier(DecisionTreeClassifier(), sparse=False) #optimized version of the CART algorithm
            #dot_data = StringIO.StringIO() 
            #tree.export_graphviz(self.classifier._clf, dot_data, feature_names=self.classifier._feature_index.keys())        
            #graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
            #graph.write_pdf("test_export_graphvix.pdf")
        
        logger.info('Training classifier')
        self.classifier.train(featuresets, labelsets, feature_index, labels, label_index, X)

