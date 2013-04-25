import argparse, os, csv
from classifiers import logger, base_path
from classifiers.naivebayes.naive_bayes_nltk import CrossValidatedNaiveBayesClassifier
from classifiers.weka import CrossValidatedWekaClassifier
from classifiers.maxent.max_ent_megam import CrossValidatedMegamMaxEntClassifier
from classifiers.scikit import CrossValidatedSciKitClassifier
from classifiers.svm.svm import CrossValidatedSVMClassifier

def get_file_name(nb, weka, megam, svmlight, sklearn):
    if weka:
        return 'resultados_weka_'+weka
    elif megam: 
        return 'resultados_megam_maxent'
    elif svmlight:
        return 'resultados_svmlight'
    elif sklearn:
        return 'resultados_sklearn_' + sklearn
    else:
        return 'resultados_nltk_nb'
    
def main(nb, weka, megam, svmlight, sklearn):
    n_folds = 5
    remove_stop_words = True    
    min_word_length = 3
    remove_duplicated_chars = False
    process_negation = True
    stem = False
    transform_lower_case = True
    remove_punctuation_marks = True
    remove_accents = True
    lemma = False
    allprepro = False

    #file_name = os.path.join(base_path, 'resultados', get_file_name(nb, weka, megam, svmlight, sklearn))
    
    #csvfile = open(file_name, 'wb')
    #res = csv.writer(csvfile, delimiter=' ',quotechar=',', quoting=csv.QUOTE_MINIMAL)    
    #res.writerow(['Fold Size','Corpus Size', 'Unigrams Pres', 'Unigrams Freq', 'Bigrams', 'Unigrams Pres + Bigrams'])
    
    #for fold_size in [50,100] + range(200, 1200, 200) + range(1400, 2600, 400):
    for fold_size in [800,1000,1400]:
        accuracies = []
        for feature in ['-u', '-wf', '-allbi', '-u -allbi']:
        #for feature in ['-u -allbi']:
            use_unigrams = (feature == '-u' or feature == '-u -bi' or feature == '-u -allbi')
            use_unigrams_frequency = (feature == '-wf')
            use_bigrams = (feature == '-bi' or feature == '-u -bi')
            use_all_bigrams = (feature == '-allbi' or feature == '-u -allbi')
            
            classifier = None
            
            if weka == 'maxent':
                classifier = CrossValidatedWekaClassifier(n_folds, fold_size, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, allprepro, 'weka.classifiers.functions.Logistic')

            elif weka == 'svm':
                classifier = CrossValidatedWekaClassifier(n_folds, fold_size, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, allprepro, 'weka.classifiers.functions.SMO')
            
            elif weka == 'tree':
                classifier = CrossValidatedWekaClassifier(n_folds, fold_size, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, allprepro, 'weka.classifiers.trees.J48')

            elif weka == 'nb':
                classifier = CrossValidatedWekaClassifier(n_folds, fold_size, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, allprepro, 'weka.classifiers.bayes.NaiveBayes')

            elif megam: 
                classifier = CrossValidatedMegamMaxEntClassifier(n_folds, fold_size, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, allprepro)

            elif svmlight:
                classifier = CrossValidatedSVMClassifier(n_folds, fold_size, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, allprepro)

            elif sklearn:
                classifier = CrossValidatedSciKitClassifier(n_folds, fold_size, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, allprepro, sklearn)

            else:
                classifier = CrossValidatedNaiveBayesClassifier(n_folds, fold_size, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, allprepro)
            
            evaluation = classifier.classify()
            accuracies.append(evaluation.get_accuracy())
            
        #res.writerow([fold_size, fold_size*n_folds*2] + accuracies)
        
    #csvfile.close
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross Validated Sentiment Classifier')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-nb', help='naive-bayes classification', action='store_true')
    group.add_argument('-weka', help='classification using WEKA API', type=str, choices=['maxent', 'svm', 'nb', 'tree'])
    group.add_argument('-megam', help='max-ent classification using MEGAM algorithm', action='store_true')
    group.add_argument('-svmlight', help='svm classification using SVMLight', action='store_true')
    group.add_argument('-sklearn', help='classification using sci-kit learn API', type=str, choices=['maxent', 'svm', 'nb', 'tree'])
    
    args = parser.parse_args()
    
    main(args.nb, args.weka, args.megam, args.svmlight, args.sklearn)
