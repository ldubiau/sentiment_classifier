# -*- coding: utf-8 -*
from collections import defaultdict

class Evaluation(object):

    def __init__(self, *k):
        super(Evaluation, self).__init__()
        self.klasses = {}
        for klass in k:
            self.klasses[klass] = defaultdict(lambda: 0)

    def add(self, expected_klass, result):
        if expected_klass == result:
            self.klasses[expected_klass]['true_positives'] += 1
            for k in set(self.klasses) - set([expected_klass]):
                self.klasses[k]['true_negatives'] += 1
        else:
            self.klasses[expected_klass]['false_negatives'] += 1
            for k in self.klasses:
                if k != expected_klass:
                    self.klasses[k]['false_positives'] += 1

    def get_precision(self, klass):
        if self.klasses[klass]['true_positives'] + self.klasses[klass]['false_positives'] == 0:
            return 0
        return float(self.klasses[klass]['true_positives']) / (self.klasses[klass]['true_positives'] + self.klasses[klass]['false_positives'])

    def get_recall(self, klass):
        return float(self.klasses[klass]['true_positives']) / (self.klasses[klass]['true_positives'] + self.klasses[klass]['false_negatives'])

    def get_f_measure(self, klass):            
        precision = self.get_precision(klass)
        recall = self.get_recall(klass)
        
        if precision + recall == 0:
            return 0
            
        return 2 * precision * recall / (precision + recall)
    
    def get_f_measure_avg(self):
        #macroavg

        f1 = 0
        for klass in self.klasses.keys():
            f1 += self.get_f_measure(klass)

        return f1 / len(self.klasses)

    def get_accuracy_avg(self):
        
        #microavg
        #klass = self.klasses.keys()[0]
        #return float(self.klasses[klass]['true_positives'] + self.klasses[klass]['true_negatives']) / self.get_cases()

        #macroavg 
        acc = 0
        for klass in self.klasses.keys():
            acc += self.get_accuracy(klass)

        return acc / len(self.klasses)
         
    def get_accuracy(self, klass):
        return float(self.klasses[klass]['true_positives']) / self.get_cases(klass)
        
    def get_cases(self, klass=None):
        if klass:
            return self.klasses[klass]['true_positives'] + self.klasses[klass]['false_negatives']
        else:
            return sum(self.klasses[k]['true_positives'] + self.klasses[k]['false_negatives'] for k in self.klasses)

    def update(self, evaluation):
        for klass in evaluation.klasses:
            for measure in evaluation.klasses[klass]:
                self.klasses[klass][measure] += evaluation.klasses[klass][measure]
