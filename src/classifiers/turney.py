# -*- coding: utf-8 -*

from classifiers.classifier import Classifier
from classifiers import logger
from classifiers.evaluation import Evaluation
from util.freeling import FreelingProcessor
from nltk import bigrams, trigrams
from math import log

class TurneyClassifier(Classifier):
    
    def __init__(self, remove_stop_words, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro):
        super(TurneyClassifier, self).__init__(remove_stop_words, min_word_length, remove_duplicated_chars, process_negation, stem, transform_lower_case, remove_punctuation_marks, remove_accents, lemma, adjectives, allprepro)
        self.freeling_processor = FreelingProcessor()

        self.pos_words = ['excelente', 'excelentes', 'bueno', 'buena', 'buenos', 'buenas', 'buenisimo', 'buenisima', 'buenisimos', 'buenisimas', 'rico', 'rica', 'ricos', 'ricas', 'espectacular', 'impecable']
        self.neg_words = ['malo', 'mala', 'mal', 'malos', 'malas', 'feo', 'fea', 'feos', 'feas', 'horrible', 'horribles', 'desastre', 'pesimo', 'pesima', 'pesimos', 'pesimas', 'mediocre', 'peor']

        #self.pos_words = ['excelente', 'excelentes', 'bueno', 'buena', 'buenos', 'buenas', 'buenisimo', 'buenisima', 'buenisimos', 'buenisimas']
        #self.neg_words = ['malo', 'mala', 'mal', 'malos', 'malas', 'feo', 'fea', 'feos', 'feas', 'horrible', 'horribles', 'desastre', 'pesimo', 'pesima', 'pesimos', 'pesimas']
        self.corpus = []
      
    def process_corpus(self):
        evaluation = Evaluation('pos', 'neg')

        self.corpus = self.pos_comments[:11000] + self.neg_comments[:11000]
        self.pos_hits = self.hits(self.pos_words)
        self.neg_hits = self.hits(self.neg_words)
        
        pos_test_corpus = self.pos_comments[:2200]
        neg_test_corpus = self.neg_comments[:2200]

        tagged_pos_test_corpus = self.tag_test_corpus(pos_test_corpus)
        tagged_neg_test_corpus = self.tag_test_corpus(neg_test_corpus)

        evaluation = self.classify_corpus(pos_test_corpus, tagged_pos_test_corpus, 'pos', evaluation)
        evaluation = self.classify_corpus(neg_test_corpus, tagged_neg_test_corpus, 'neg', evaluation)

        logger.info('Total TestSet Size: {} - Avg Accuracy: {}'.format(evaluation.get_cases(), evaluation.get_accuracy()))
        return evaluation

    def classify_corpus(self, test_corpus, tagged_test_corpus, expected_klass, evaluation):
        i = 0
        for doc in tagged_test_corpus:
            print "Document {}: [{}]".format(i, ','.join(x.encode('utf-8') for x in test_corpus[i]))
            print "Tagged Document {}: [{}]".format(i, ','.join(x.word.encode('utf-8') for x in doc))
             
            klass = self.classify_comment(doc)
            print "klass: {} - expected_klass: {}\n\n".format(klass, expected_klass)
            evaluation.add(expected_klass, klass)
            i = i + 1
        return evaluation

    def tag_test_corpus(self, test_corpus):
        full_text = ''
        for doc in test_corpus:
            doc_text = ' '.join(doc)
            doc_text = doc_text.replace('|', '')
            full_text += doc_text + '|\n'

        #print full_text.encode('utf-8')
        freeling_docs = self.freeling_processor.process_text(full_text)

        i = 0
        tagged_test_corpus = []
        for doc in test_corpus:
            tagged_test_corpus.append(freeling_docs[i])
            i = i + 1
            
        return tagged_test_corpus

    def get_semantic_orientation(self, bigram):
        near_pos = self.near(bigram, self.pos_words) + 0.01
        near_neg = self.near(bigram, self.neg_words) + 0.01

        so = float(near_pos)*self.neg_hits / near_neg / self.pos_hits
     
        print "bigram: " + str(bigram) 
        print "near_pos: " + str(near_pos)
        print "near_neg: " + str(near_neg)
        print "neg_hits: " + str(self.neg_hits)
        print "pos_hits: " + str(self.pos_hits)
        print "so: " + str(log(so) / log(2))
        return log(so) / log(2)
 
    def near(self, bigram, hits_words):
        near = 0
        for doc in self.corpus:
            if bigram in bigrams(doc):
                if len(set(doc).intersection(hits_words)):
                    near += 1

        return near

    def hits(self, hits_words):
        hits = 0
        for doc in self.corpus:
            hit = False
            for word in doc:
                if word.lower() in hits_words:
                    hit = True
            if hit: hits = hits + 1 
        return hits

    def matches_opinion_patterns(self, tagged_ngram):
        patterns = [('AQ', 'NC', ' '), 
                    ('NC', 'AQ', 'NC'),
                    ('R', 'AQ', 'NC'),
                    ('R', 'V', ' '),
                    ('V', 'R', ' ')]

        for p in patterns:
            match_word1 = tagged_ngram[0].tag.startswith(p[0])
            match_word2 = tagged_ngram[1].tag.startswith(p[1])
            match_word3 = False

            if len(tagged_ngram) > 2:
                match_word3 = tagged_ngram[2].tag.startswith(p[2])

            if match_word1 and match_word2 and not match_word3:
                return True

        return False

    def classify_comment(self,doc):
        so = 0
        if len(doc) >= 3:
            ngrams = trigrams(doc)
        else:
            ngrams = bigrams(doc)

        for tagged_ngram in ngrams:
            if self.matches_opinion_patterns(tagged_ngram):
                bigram = (tagged_ngram[0].word, tagged_ngram[1].word)
                b_so = self.get_semantic_orientation(bigram)
                so += b_so
                #print "bigram: " + tagged_trigram[0].word + " " + tagged_trigram[1].word + " - bigram so: " + str(b_so)
        #print "so: " + str(so)

        if so > 0:
            return 'pos' 
        return 'neg'

