# -*- coding: utf-8 -*
import argparse
import nltk, os

from classifiers.classifier_ld import SupervisedClassifier
from classifiers.evaluation import Evaluation
from classifiers import logger
from collections import defaultdict
from nltk.probability import FreqDist, ELEProbDist
from nltk.classify import NaiveBayesClassifier

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class CrossValidatedNaiveBayesClassifier(SupervisedClassifier):
    
    def __init__(self, n_folds, fold_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, out_of_domain_test, proportion_of_positives):
        super(CrossValidatedNaiveBayesClassifier, self).__init__(n_folds, fold_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, out_of_domain_test, proportion_of_positives)
        self.folds_label_freqdist = defaultdict()
        self.folds_feature_freqdist = defaultdict()
        self.folds_feature_values = defaultdict()
        self.folds_fnames = defaultdict()

    def get_classifier(self, test_fold, estimator=ELEProbDist):
        label_freqdist = FreqDist()
        feature_freqdist = defaultdict(FreqDist)
        feature_values = defaultdict(set)
        fnames = set()
        
        folds = self.folds_label_freqdist.keys()
        
        for fold in folds:
            if fold != test_fold:
                label_freqdist = label_freqdist + self.folds_label_freqdist[fold]                
                fnames.update(self.folds_fnames[fold])
                
                for (fname, values) in self.folds_feature_values[fold].items():
                    feature_values[fname].update(values)
                
        for fold in folds:
            if fold != test_fold:
                for ((label, fname), freqdist) in self.folds_feature_freqdist[fold].items():
                    for val in feature_values[fname]:
                        feature_freqdist[label, fname].inc(val, freqdist[val])
        
        # Create the P(label) distribution
        label_probdist = estimator(label_freqdist)

        # Create the P(fval|label, fname) distribution
        feature_probdist = {}
        for ((label, fname), freqdist) in feature_freqdist.items():
            probdist = estimator(freqdist, bins=len(feature_values[fname]))
            feature_probdist[label,fname] = probdist

        return NaiveBayesClassifier(label_probdist, feature_probdist)
        
    def classify_comments(self, test_fold, test_comments):
        self.classifier = self.get_classifier(test_fold)
        logger.info(self.classifier.show_most_informative_features())

        evaluation = Evaluation('pos', 'neg')
        for comment, expected_klass in test_comments:
            klass = self.classifier.classify(comment)
            #if klass != expected_klass:
            #print 'expected class: %s, class: %s, comment: %s' %(expected_klass, klass, " ".join(comment))
            evaluation.add(expected_klass, klass)
        return evaluation

    def build_nltk_freq_distributions(self, fold, labeled_featuresets):        
        label_freqdist = FreqDist()
        feature_freqdist = defaultdict(FreqDist)
        feature_values = defaultdict(set)
        fnames = set()

        for featureset, label in labeled_featuresets:
            label_freqdist.inc(label)
            for fname, fval in featureset.items():
                feature_freqdist[label, fname].inc(fval)
                feature_values[fname].add(fval)
                fnames.add(fname)

        for label in label_freqdist:
            num_samples = label_freqdist[label]
            for fname in fnames:
                count = feature_freqdist[label, fname].N()
                feature_freqdist[label, fname].inc(None, num_samples-count)
                feature_values[fname].add(None)
    
        self.folds_label_freqdist[fold] = label_freqdist
        self.folds_feature_freqdist[fold] = feature_freqdist
        self.folds_feature_values[fold] = feature_values
        self.folds_fnames[fold] = fnames
        
def main(n_folds=3, fold_size=100, remove_stop_words=False, use_unigrams=False, use_unigrams_frequency=False, use_bigrams=False, min_word_length=None, remove_duplicated_chars=False, process_negation=False, stem=False):
    assert n_folds > 1

    classifier = CrossValidatedNaiveBayesClassifier(n_folds, fold_size, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem)
    classifier.classify()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process comments.')
    parser.add_argument('-f', help='Number of folds', type=int, default=3)
    parser.add_argument('-s', help='Fold size', type=int)
    parser.add_argument('-sw', help='Remove Stop Words', action='store_true')
    parser.add_argument('-u', help='Use unigrams feature extractor', action='store_true')
    parser.add_argument('-wf', help='Use unigrams frequency feature extractor', action='store_true')
    parser.add_argument('-bi', help='Use bigrams feature extractor', action='store_true')
    parser.add_argument('-wl', help='Minimum word length', type=int)
    parser.add_argument('-dc', help='Remove Duplicated Characters', action='store_true')
    parser.add_argument('-neg', help='Process Negation', action='store_true')
    parser.add_argument('-stem', help='Stem', action='store_true')
    args = parser.parse_args()
    main(args.f, args.s, args.sw, args.u, args.wf, args.bi, args.wl, args.dc, args.neg, args.stem)
