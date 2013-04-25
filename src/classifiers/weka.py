import nltk
import os
import tempfile
import subprocess

from classifiers import base_path
from classifiers.classifier import SupervisedClassifier
from classifiers.evaluation import Evaluation
from classifiers import logger
from collections import defaultdict
from nltk.classify.weka import WekaClassifier
from nltk.classify import weka
from nltk.internals import java, config_java

weka_classpath = os.path.join(base_path, 'externaltools', 'weka.jar')

class CrossValidatedWekaClassifier(SupervisedClassifier, WekaClassifier):

    def __init__(self, n_folds, fold_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, javaclass):
        super(CrossValidatedWekaClassifier, self).__init__(n_folds, fold_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro)
        self.features_names = []
        self.labels = ['neg', 'pos']
        self.javaclass = javaclass
        self.train_filename = ''
        self.test_filename = ''
        
    def train(self, training_documents, feature_extractor):
        logger.info('Creating training dataset, documents size {}'.format(len(training_documents)))
        training_set = nltk.classify.util.apply_features(feature_extractor.extract, training_documents)
        
        logger.info('Extracting features')
        self.features_names = sorted(feature_extractor.extract_features_names())
        
        temp_dir = tempfile.mkdtemp()
        self.train_filename = os.path.join(temp_dir, 'train.arff')               
        
        logger.info('Writing Training WEKA File: ' + self.train_filename)
        self._write_ARFF_file(self.train_filename, training_set)
        
    def _get_ARFF_header(self):
        # Relation name
        s = '@RELATION rel\n\n'

        # Input attribute specifications
        for fname, ftype in self.features_names:
            s += '@ATTRIBUTE %-30r %s\n' % (fname, ftype)

        # Label attribute specification
        s += '@ATTRIBUTE %-30r {%s}\n' % ('-label-', ','.join(self.labels))
        
        return s
        
    def _write_ARFF_file(self, train_filename, training_set):
        outfile = open(train_filename, 'w')
        try:
            outfile.write(self._get_ARFF_header())            
            outfile.write('\n@DATA\n')
            
            for (document_features, label) in training_set:
                for fname, ftype in self.features_names:
                    outfile.write('%s,' % document_features.get(fname))
                outfile.write('%s\n' % label)

        finally:
            outfile.close()
        
    def _classify_using_weka(self, test_comments, feature_extractor):
        test_set = nltk.classify.util.apply_features(feature_extractor.extract, test_comments)
        
        temp_dir = tempfile.mkdtemp()
        self.test_filename = os.path.join(temp_dir, 'test.arff')               
        
        logger.info('Writing Test WEKA File: ' + self.test_filename)
        self._write_ARFF_file(self.test_filename, test_set)

        cmd = [self.javaclass, '-t', self.train_filename, '-T', self.test_filename] + ['-p', '0']
        
        logger.info('Executing WEKA: ' + str(cmd))
        
        config_java(options='-Xmx2000M')
        (stdout, stderr) = java(cmd, classpath=weka_classpath,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        
        return self.parse_weka_output(stdout.split('\n'))
        
    def classify_comments(self, test_comments, feature_extractor):
        evaluation = Evaluation('pos', 'neg')        
        
        klasses = self._classify_using_weka(test_comments, feature_extractor)        
        i = 0
        for test_comment, expected_klass in test_comments:
            evaluation.add(expected_klass, klasses[i])
            i += 1
        return evaluation
