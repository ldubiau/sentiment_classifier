import nltk, os

from classifiers import base_path
from collections import defaultdict
from nltk.classify.maxent import MaxentClassifier
from nltk.classify import megam
from classifiers.extractors import BigramFeatureExtractor

megam_path = os.path.join(base_path, 'externaltools')

def main():
    megam.config_megam(megam_path)

    feature_extractor = BigramFeatureExtractor()
    training_documents = []
    training_documents.append((['muy', 'buena', 'comida', 'hola'], 'pos'))
    training_documents.append((['muy', 'mala', 'comida'], 'neg'))
    training_documents.append((['muy', 'mala', 'comida', 'hola'], 'neg'))
    training_documents.append((['buena', 'comida'], 'pos'))

    training_set = nltk.classify.util.apply_features(feature_extractor.extract, training_documents)
    classifier = nltk.MaxentClassifier.train(training_set, algorithm='megam', explicit=False, bernoulli=True, model='binary')
    classifier.show_most_informative_features()

if __name__ == '__main__':
    main()