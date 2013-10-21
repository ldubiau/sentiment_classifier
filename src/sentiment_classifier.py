# -*- coding: utf-8 -*
import argparse
from classifiers import logger
from decimal import *
from classifiers.naivebayes.naive_bayes_nltk import CrossValidatedNaiveBayesClassifier
from classifiers.weka import CrossValidatedWekaClassifier
from classifiers.maxent.max_ent_megam import CrossValidatedMegamMaxEntClassifier
from classifiers.scikit_opt import CrossValidatedSciKitClassifier
#from classifiers.svm.svm import CrossValidatedSVMClassifier
from classifiers.turney import TurneyClassifier

def main(nb=True, weka=None, megam=False, svmlight=False, sklearn=None, turney=False,n_folds=5, corpus_size=100, fold_number = None, remove_stop_words=False, use_unigrams=False, use_unigrams_frequency=False, use_bigrams=False, use_all_bigrams = False, min_word_length=None, remove_duplicated_chars=False, process_negation=False, stem=False, transform_lower_case=False, remove_punctuation_marks=False, remove_accents=False, lemma=False, adjectives=False, allprepro=False, out_of_domain_test=False, proportion_of_positives=0.5):
    if weka == 'maxent':
        classifier = CrossValidatedWekaClassifier(n_folds, corpus_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, out_of_domain_test, proportion_of_positives, 'weka.classifiers.functions.Logistic')

    elif weka == 'svm':
        classifier = CrossValidatedWekaClassifier(n_folds, corpus_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, out_of_domain_test, proportion_of_positives, 'weka.classifiers.functions.SMO')
    
    elif weka == 'tree': #C4.5 algorithm or use weka.classifiers.trees.SimpleCart
        classifier = CrossValidatedWekaClassifier(n_folds, corpus_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, out_of_domain_test, proportion_of_positives, 'weka.classifiers.trees.J48')

    elif weka == 'nb':
        classifier = CrossValidatedWekaClassifier(n_folds, corpus_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, out_of_domain_test, proportion_of_positives, 'weka.classifiers.bayes.NaiveBayes')

    elif megam: 
        classifier = CrossValidatedMegamMaxEntClassifier(n_folds, corpus_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, out_of_domain_test, proportion_of_positives)

    elif svmlight:
        classifier = CrossValidatedSVMClassifier(n_folds, corpus_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, out_of_domain_test, proportion_of_positives)

    elif sklearn:
        classifier = CrossValidatedSciKitClassifier(n_folds, corpus_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, out_of_domain_test, proportion_of_positives, sklearn)

    elif nb:
        print proportion_of_positives
        classifier = CrossValidatedNaiveBayesClassifier(n_folds, corpus_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, out_of_domain_test, proportion_of_positives)

    elif turney:
        classifier = TurneyClassifier(corpus_size, remove_stop_words, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, out_of_domain_test, proportion_of_positives)

    classifier.classify()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentiment Classification Tool')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-nb', help='Naive-bayes classification', action='store_true')
    group.add_argument('-weka', help='Classification using WEKA API', type=str, choices=['maxent', 'svm', 'nb', 'tree'])
    group.add_argument('-megam', help='MaxEnt classification using MEGAM algorithm', action='store_true')
    group.add_argument('-svmlight', help='SVM classification using SVMLight', action='store_true')
    group.add_argument('-sklearn', help='Classification using sci-kit learn API', type=str, choices=['maxent', 'svm', 'nb', 'tree'])
    group.add_argument('-turney', help='Classification using Turney algorithm', action='store_true')

    parser.add_argument('-f', help='Number of folds for supervised algorithms using k-fold cross validation. If this parameter is not provided then holdout validation is performed.', type=int)
    parser.add_argument('-fn', help='Fold number for supervised algorithms using k-fold cross validation', type=int)
    parser.add_argument('-s', help='Corpus size', type=int)
    parser.add_argument('-od', help='Out of domain testing', action='store_true')
    parser.add_argument('-u', help='Use top training unigrams as feature extractor', action='store_true')
    parser.add_argument('-wf', help='Use top training unigrams frequency as feature extractor', action='store_true')
    parser.add_argument('-docbi', help='Use document bigrams as feature extractor', action='store_true')
    parser.add_argument('-bi', help='Use top training bigrams as feature extractor', action='store_true')
    parser.add_argument('-sw', help='Remove stop words', action='store_true')
    parser.add_argument('-wl', help='Filter words by minimum length', type=int)
    parser.add_argument('-dc', help='Remove duplicated characters', action='store_true')
    parser.add_argument('-neg', help='Preprocess negations', action='store_true')
    parser.add_argument('-stem', help='Use stemmed words', action='store_true')
    parser.add_argument('-lc', help='Transform chars to lower case', action='store_true')
    parser.add_argument('-punct', help='Remove punctuation marks', action='store_true')
    parser.add_argument('-acc', help='Remove spanish accents', action='store_true')
    parser.add_argument('-lemma', help='Use lemmatized words', action='store_true')
    parser.add_argument('-adj', help='Use just adjectives', action='store_true')
    parser.add_argument('-allprepro', help='Apply all preprocessors', action='store_true')
    parser.add_argument('-pp', help='Proportion of positive comments for unbalanced experiences', type=Decimal, default=0.5)
    args = parser.parse_args()
    
    logger.info('Starting Sentiment Analysis Process. Params: ' + str(args))
    main(args.nb, args.weka, args.megam, args.svmlight, args.sklearn, args.turney,
        args.f, args.s, args.fn, args.sw, args.u, args.wf, args.docbi, args.bi, 
        args.wl, args.dc, args.neg, args.stem, args.lc, args.punct, args.acc,
        args.lemma, args.adj, args.allprepro, args.od, args.pp)

