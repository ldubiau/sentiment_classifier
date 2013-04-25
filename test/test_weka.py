import nltk, os

from classifiers import base_path
from collections import defaultdict
from nltk.classify.weka import WekaClassifier
from nltk.classify import weka
from classifiers.extractors import BigramFeatureExtractor

weka_path = os.path.join(base_path, 'externaltools', 'weka.jar')

def main():
    weka.config_weka(weka_path)

    feature_extractor = BigramFeatureExtractor()
    training_documents = []
    training_documents.append((['muy', 'buena', 'comida', 'hola'], 'pos'))
    training_documents.append((['muy', 'mala', 'comida'], 'neg'))
    training_documents.append((['muy', 'mala', 'comida', 'hola'], 'neg'))
    training_documents.append((['buena', 'comida'], 'pos'))

    test_comment = "buena"

    training_set = nltk.classify.util.apply_features(feature_extractor.extract, training_documents)
    options = []
    options.append('-no-cv')
    classifier = nltk.WekaClassifier.train('weka.model', training_set, 'log_regression', options)
    print classifier.classify(feature_extractor.extract(test_comment))

if __name__ == '__main__':
    main()
