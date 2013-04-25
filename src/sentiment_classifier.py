# -*- coding: utf-8 -*
import argparse
from classifiers import logger

from classifiers.naivebayes.naive_bayes_nltk import CrossValidatedNaiveBayesClassifier
from classifiers.weka import CrossValidatedWekaClassifier
from classifiers.maxent.max_ent_megam import CrossValidatedMegamMaxEntClassifier
from classifiers.scikit_opt import CrossValidatedSciKitClassifier
from classifiers.svm.svm import CrossValidatedSVMClassifier
from classifiers.turney import TurneyClassifier

def main(nb=True, weka=None, megam=False, svmlight=False, sklearn=None, turney=False,n_folds=5, fold_size=100, fold_number = None, remove_stop_words=False, use_unigrams=False, use_unigrams_frequency=False, use_bigrams=False, use_all_bigrams = False, min_word_length=None, remove_duplicated_chars=False, process_negation=False, stem=False, transform_lower_case=False, remove_punctuation_marks=False, remove_accents=False, lemma=False, adjectives=False, allprepro=False):
    assert n_folds > 1
    
    if weka == 'maxent':
        classifier = CrossValidatedWekaClassifier(n_folds, fold_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, 'weka.classifiers.functions.Logistic')

    elif weka == 'svm':
        classifier = CrossValidatedWekaClassifier(n_folds, fold_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, 'weka.classifiers.functions.SMO')
    
    elif weka == 'tree': #C4.5 algorithm or use weka.classifiers.trees.SimpleCart
        classifier = CrossValidatedWekaClassifier(n_folds, fold_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, 'weka.classifiers.trees.J48')

    elif weka == 'nb':
        classifier = CrossValidatedWekaClassifier(n_folds, fold_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, 'weka.classifiers.bayes.NaiveBayes')

    elif megam: 
        classifier = CrossValidatedMegamMaxEntClassifier(n_folds, fold_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro)

    elif svmlight:
        classifier = CrossValidatedSVMClassifier(n_folds, fold_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro)

    elif sklearn:
        classifier = CrossValidatedSciKitClassifier(n_folds, fold_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro, sklearn)

    elif nb:
        classifier = CrossValidatedNaiveBayesClassifier(n_folds, fold_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro)

    elif turney:
        classifier = TurneyClassifier(remove_stop_words, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro)

    classifier.classify()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross Validated Sentiment Classifier')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-nb', help='naive-bayes classification', action='store_true')
    group.add_argument('-weka', help='classification using WEKA API', type=str, choices=['maxent', 'svm', 'nb', 'tree'])
    group.add_argument('-megam', help='max-ent classification using MEGAM algorithm', action='store_true')
    group.add_argument('-svmlight', help='svm classification using SVMLight', action='store_true')
    group.add_argument('-sklearn', help='classification using sci-kit learn API', type=str, choices=['maxent', 'svm', 'nb', 'tree'])
    group.add_argument('-turney', help='classification using Turney algorithm', action='store_true')

    parser.add_argument('-f', help='number of folds', type=int, default=5)
    parser.add_argument('-s', help='fold size', type=int)
    parser.add_argument('-fn', help='fold number', type=int)
    parser.add_argument('-u', help='use top training unigrams feature extractor', action='store_true')
    parser.add_argument('-wf', help='use top training unigrams frequency feature extractor', action='store_true')
    parser.add_argument('-docbi', help='use document bigrams feature extractor', action='store_true')
    parser.add_argument('-bi', help='use top training bigrams feature extractor', action='store_true')
    parser.add_argument('-sw', help='remove stop words', action='store_true')
    parser.add_argument('-wl', help='filter words by minimum length', type=int)
    parser.add_argument('-dc', help='remove duplicated characters', action='store_true')
    parser.add_argument('-neg', help='preprocess negation', action='store_true')
    parser.add_argument('-stem', help='use stemmed words', action='store_true')
    parser.add_argument('-lc', help='transform to lower case', action='store_true')
    parser.add_argument('-punct', help='remove punctuation marks', action='store_true')
    parser.add_argument('-acc', help='transform accented letters', action='store_true')
    parser.add_argument('-lemma', help='use lemmatized words', action='store_true')
    parser.add_argument('-adj', help='use just adjectives', action='store_true')
    parser.add_argument('-allprepro', help='apply all preprocessors', action='store_true')
    args = parser.parse_args()
    
    logger.info('Starting Sentiment Analysis Process. Params: ' + str(args))
    main(args.nb, args.weka, args.megam, args.svmlight, args.sklearn, args.turney,
        args.f, args.s, args.fn, args.sw, args.u, args.wf, args.docbi, args.bi, 
        args.wl, args.dc, args.neg, args.stem, args.lc, args.punct, args.acc,
        args.lemma, args.adj, args.allprepro)

