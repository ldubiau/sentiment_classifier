from classifiers.naivebayes.naive_bayes_nltk import CrossValidatedNaiveBayesClassifier

def main():
    n_folds = 5
    sizes = (50, 400, 1000, 2200)
    use_unigrams = True
    use_unigrams_frequency = False
    use_bigrams = False
    use_all_bigrams = False
    adjectives = False
     
    for fold_size in sizes:
        for prepro in ('None', '-sw', '-neg', '-wl 3', '-dc', '-stem', '-lc', '-punct', '-acc', '-lemma', '-allprepro', '-selected'):
            print "Fold size: %d, Preproceso: %s" %(fold_size, prepro)
            remove_stop_words = prepro == '-sw' or prepro == '-selected'
            if prepro == '-wl 3' or prepro == '-selected':
                min_word_length = 3 
            else: 
                min_word_length = None
            remove_duplicated_chars = prepro == '-dc'
            process_negation = prepro == '-neg' or prepro == '-selected'
            stem = prepro == '-stem'
            transform_lower_case = prepro == '-lc' or prepro == '-selected'
            remove_punctuation_marks = prepro == '-punct' or prepro == '-selected'
            remove_accents = prepro == '-acc' or prepro == '-selected'
            lemma = prepro == '-lemma'
            allprepro = prepro == '-allprepro'
          
            classifier = CrossValidatedNaiveBayesClassifier(n_folds, fold_size, fold_number, remove_stop_words, use_unigrams, use_unigrams_frequency, use_bigrams, use_all_bigrams, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro)
            classifier.classify()

if __name__ == '__main__':
    main()
