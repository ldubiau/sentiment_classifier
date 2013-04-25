# -*- coding: utf-8 -*
from classifiers.evaluation import Evaluation
import pytest

__author__ = 'Luciana Dubiau'

data = [(('pos', 'pos'), ('pos', 'pos'), ('pos', 'neg'), ('neg', 'neg'), ('neg', 'neg'), ('neg', 'neg'), ('neg', 'pos'), ('neg', 'pos')),
        (('pos', 'pos'), ('pos', 'pos'), ('pos', 'pos'), ('pos', 'neg'), ('neg', 'neg'), ('neg', 'pos'), ('neg', 'pos'), ('neg', 'pos'), ('neg', 'pos'))]

def count(cases, expected_klass, actual_klass):
    return len([1 for expected, actual in cases if expected == expected_klass and actual == actual_klass])

@pytest.mark.parametrize(('cases'), data)
def test_evaluation(cases):
    e = Evaluation('pos', 'neg')
    for expected, actual in cases:
        e.add(expected, actual)
    tp_pos = count(cases, 'pos', 'pos')
    fp_pos = count(cases, 'neg', 'pos')
    tn_pos = count(cases, 'neg', 'neg')
    fn_pos = count(cases, 'pos', 'neg')
    tp_neg = count(cases, 'neg', 'neg')
    fp_neg = count(cases, 'pos', 'neg')
    tn_neg = count(cases, 'pos', 'pos')
    fn_neg = count(cases, 'neg', 'pos')
    assert e.klasses['pos']['true_positives'] == tp_pos
    assert e.klasses['pos']['false_positives'] == fp_pos
    assert e.klasses['pos']['true_negatives'] == tn_pos
    assert e.klasses['pos']['false_negatives'] == fn_pos
    assert e.klasses['neg']['true_positives'] == tp_neg
    assert e.klasses['neg']['false_positives'] == fp_neg
    assert e.klasses['neg']['true_negatives'] == tn_neg
    assert e.klasses['neg']['false_negatives'] == fn_neg
    precision_pos = float(tp_pos) / (tp_pos + fp_pos)
    assert e.get_precision('pos') == precision_pos
    precision_neg = float(tp_neg) / (tp_neg + fp_neg)
    assert e.get_precision('neg') == precision_neg
    recall_pos = float(tp_pos) / (tp_pos + fn_pos)
    assert e.get_recall('pos') == recall_pos
    recall_neg = float(tp_neg) / (tp_neg + fn_neg)
    assert e.get_recall('neg') == recall_neg
    assert e.get_f_measure('pos') == 2.0 * precision_pos * recall_pos / (precision_pos + recall_pos)
    assert e.get_f_measure('neg') == 2.0 * precision_neg * recall_neg / (precision_neg + recall_neg)
    assert e.get_accuracy() == float(tp_pos + tn_pos) / (tp_pos + tn_pos + fp_pos + fn_pos)
