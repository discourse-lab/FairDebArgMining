"""
Helper scripts related to eval
"""
import statistics

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import pickle
import sys

np.set_printoptions(suppress=True)


def evaluate_classification(Ypred, Ygold, average='macro'):
    """
    Classification eval on test set
    Ypred, Ygold must be list of the same length
    if averaged: only return accuracy and macro prec/rec/f1/support across all classes
    else: return accuracy and more detailed prec/rec etc. for each class + conf_mat
    """
    acc = accuracy_score(y_true=Ygold, y_pred=Ypred)
    if average is not None:
        prfs = precision_recall_fscore_support(y_true=Ygold, y_pred=Ypred, average=average)
    else:
        prfs = precision_recall_fscore_support(y_true=Ygold, y_pred=Ypred)
    return acc, prfs


def print_avg_prfs(prfs_list):
    """
    print average prec, rec, f1 across k folds
    prfs_list = list(tuple) where each tuple is the prfs output for one fold
    return: [precs, recs, f1s]
    """
    prfs = np.array(prfs_list)
    assert prfs.shape == (len(prfs_list), 4)
    # disregard the last col, i.e. the support values
    prf = prfs[:,:3].astype(float)
    metrics = np.mean(prf, axis=0)
    metrics = np.around(metrics, decimals=3).tolist()
    return metrics


def print_avg_prfs_by_class(prfs_list_by_class, num_folds, num_classes):
    """
    print average prec, rec, f1 by class across k folds
    :param prfs_list_by_class list(tuple(array([prec,...]), tuple(array([rec,...]), tuple(array([f1,...], tuple(array([support,...]))
    :param num_folds number of folds
    :param num_classes number of classes
    :return nd.array of shape (4, num_classes)
    """
    all_metrics = np.array(prfs_list_by_class)
    assert all_metrics.shape == (num_folds, 4, num_classes)
    return np.round(np.mean(all_metrics, axis=0), decimals=3)


def print_errors(test_X, preds, golds, index2lab=None):
    """
    Print out sentences in the test / val set for which model has made a prediction error
    :param test_X: list of input test sents
    :param preds: list of model predictions
    :param golds: list of gold labels
    All 3 must be index-aligned
    :param index2lab: mapping from index to label name (optional)
    """
    print('Sent\tPred\tGold')
    for sent, pred, gold in zip(test_X, preds, golds):
        if pred != gold:
            if index2lab is not None:
                print(sent + '\t' + index2lab[pred] + '\t' + index2lab[gold])
            else:
                print(sent + '\t' + pred + '\t' + gold)
    return


def full_eval_pickled(model_out_dir, num_folds=5):
    prfs_class_list = []
    prfs_macro_list = []
    accs_list = []
    for i in range(num_folds):
        data = pickle.load(open('{}/fold{}.p'.format(model_out_dir, i + 1), 'rb'))
        ypreds = data[0]
        ygolds = data[1]

        print('Results for Fold', i + 1)
        acc, prfs = evaluate_classification(ypreds, ygolds, average='macro')
        prfs_macro_list.append(prfs)
        accs_list.append(acc)
        print('Acc:', round(acc, 3))
        print('PRFS:', prfs)
        print('-'*50)

        # by class
        _, prfs_class = evaluate_classification(ypreds, ygolds, average=None)
        prfs_class_list.append(prfs_class)

    print(f'Eval across {num_folds} folds:')
    print('Accuracy', round(statistics.mean(accs_list), 3))
    print('Precision, Recall, F1')
    print(print_avg_prfs(prfs_macro_list))

    print(f'PRF by class across {num_folds} folds: ')
    print(print_avg_prfs_by_class(prfs_class_list, num_folds=num_folds, num_classes=6))
