# -*- coding: utf-8 -*
import re
import nltk
from nltk.stem.snowball import SpanishStemmer
from util.freeling import FreelingProcessor
from classifiers import logger

__author__ = 'Luciana Dubiau'

DEFAULT_PUNCTUATION = ('\r', '\n', '.', ',', ':', ';', '-', '(', ')', '!', '?')
DEFAULT_NEGATION_WORDS = ('no', )
NEGATION_PREFIX = 'NOT_'
DEFAULT_TRANSLATION = (
    (u'á', 'a'), (u'é', 'e'), (u'í', 'i'), (u'ó', 'o'), (u'ú', 'u'), (u'à', 'a'), (u'è', 'e'), (u'ì', 'i'), (u'ò', 'o'),
    (u'ù', 'u'))


class CorpusProcessor(object):
    def process(self, corpus):
        pass


class DocumentAtATimeCorpusProcessor(CorpusProcessor):
    def process(self, corpus):
        return [self.process_document(document) for document in corpus]

    def process_document(self, document):
        pass


class CompositeCorpusProcessor(CorpusProcessor):
    def __init__(self, processors):
        super(CompositeCorpusProcessor, self).__init__()
        self.processors = processors

    def process(self, corpus):
        result = corpus
        for processor in self.processors:
            result = processor.process(result)
        return result


class FreeLingProcessor(CorpusProcessor):
    def __init__(self, filter=lambda term: True):
        super(FreeLingProcessor, self).__init__()
        self.freeling_processor = FreelingProcessor()
        self.filter = filter

    def process(self, corpus):
        full_text = ''
        for doc in corpus:
            doc_text = ' '.join(doc)
            doc_text = doc_text.replace('|', '')
            full_text += doc_text + '|\n'
            
        freeling_docs = self.freeling_processor.process_text(full_text)
            
        i = 0
        processed_corpus = []
        for doc in corpus:
            logger.info("original doc: [" + ' '.join(doc) + "]")
            logger.info("processed doc: [" + ' '.join("u'" + x.word + "'" for x in freeling_docs[i]) + "]")
            processed_corpus.append([self.extract_feature(term) for term in freeling_docs[i] if
                                     self.filter(term)])            
            i += 1
        return processed_corpus

    def extract_feature(self, term):
        return term.lemma


class TokenizerProcessor(DocumentAtATimeCorpusProcessor):
    def process_document(self, document):
        return nltk.wordpunct_tokenize(document)


class LowerCaseProcessor(DocumentAtATimeCorpusProcessor):
    def process_document(self, document):
        return [word.lower() for word in document]


class TransformCharactersProcessor(DocumentAtATimeCorpusProcessor):
    def __init__(self, translation=DEFAULT_TRANSLATION):
        super(TransformCharactersProcessor, self).__init__()
        self.translation = translation

    def process_document(self, document):
        processed_document = []
        for word in document:
            for old_char, new_char in self.translation:
                word = re.sub(old_char, new_char, word)
            processed_document.append(word)
        return processed_document


class FilterProcessor(DocumentAtATimeCorpusProcessor):
    def __init__(self):
        super(FilterProcessor, self).__init__()

    def process_document(self, document):
        return [word for word in document if self.__word_filter__(word)]

    def __word_filter__(self, word):
        pass


class FilterPunctuationProcessor(FilterProcessor):
    def __init__(self, punctuation=DEFAULT_PUNCTUATION):
        super(FilterPunctuationProcessor, self).__init__()
        self.punctuation = punctuation

    def __word_filter__(self, word):
        #match = re.match(ur'^[A-Za-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00c1\u00c9\u00cd\u00d3\u00da\u00f1\u00d1_]*$', word)
        #return match
        
        return word not in DEFAULT_PUNCTUATION


class FilterStopWordsProcessor(FilterProcessor):
    def __init__(self, negation_words=DEFAULT_NEGATION_WORDS):
        super(FilterStopWordsProcessor, self).__init__()
        self.stopwords = tuple([word.decode('utf-8').lower() for word in nltk.corpus.stopwords.words('spanish')])
        self.stopwords = TransformCharactersProcessor().process_document(self.stopwords)
        self.stopwords = tuple(set(self.stopwords) - set(negation_words))

    def __word_filter__(self, word):
        return word not in self.stopwords


class FilterWordLengthProcessor(FilterProcessor):
    def __init__(self, min_length=3, skip_words=DEFAULT_NEGATION_WORDS + DEFAULT_PUNCTUATION):
        super(FilterWordLengthProcessor, self).__init__()
        self.min_length = min_length
        self.skip_words = skip_words

    def __word_filter__(self, word):
        return word in self.skip_words or len(word) >= self.min_length


class TransformDuplicatedCharsProcessor(DocumentAtATimeCorpusProcessor):
    def __init__(self):
        super(TransformDuplicatedCharsProcessor, self).__init__()

    def process_document(self, document):
        return [re.sub(r'([a-z!])\1\1+', r'\1', word) for word in document]


class TransformNegativeWordsProcessor(DocumentAtATimeCorpusProcessor):
    def __init__(self, negation_words=DEFAULT_NEGATION_WORDS, punctuation=DEFAULT_PUNCTUATION):
        super(TransformNegativeWordsProcessor, self).__init__()
        self.punctuation = punctuation
        self.negation_words = negation_words

    def process_document(self, document):
        negate = False
        processed_document = []
        for word in document:
            if word in self.negation_words:
                negate = True
            elif word in self.punctuation:
                negate = False

            if word not in self.negation_words:
                if negate:
                    processed_document.append(NEGATION_PREFIX + word)
                else:
                    processed_document.append(word)

        return processed_document


class StemmerProcessor(DocumentAtATimeCorpusProcessor):
    def __init__(self):
        super(StemmerProcessor, self).__init__()
        self.stemmer = SpanishStemmer()

    def process_document(self, document):
        processed_document = []
        for word in document:
            processed_document.append(self.stemmer.stem(word))
        return processed_document
