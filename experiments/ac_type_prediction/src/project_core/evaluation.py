"""
Helper functions related to eval of LLM-based models
"""

import pickle
import sys
import statistics
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import llm_helper
import llm_eval


def map_gold_labels(label, mapping):
    """ Map a single label from gold annotation to 5cl label set """
    return mapping[label] if label in mapping else label


def read_pred(llm_out_file):
    """
    Read in pickled files of LLM output
    """
    fi = open(llm_out_file, 'rb')
    llm_out = pickle.load(fi)
    fi.close()
    return llm_out


def read_gold(dir_to_essay_files, demo_essays=None, distribution_file=None):
    """
    Reading in gold essay files
    :return
    """
    eval_data, _ = llm_helper.get_data_all_essays(
        dir_to_essay_files=dir_to_essay_files,
        file_format='txt',
        demo_essays=demo_essays
    )
    if distribution_file:
        all_const = llm_helper.get_all_essay_const(
            dir_to_essay_files=dir_to_essay_files,
            distribution_file=distribution_file,
            demo_essays=demo_essays
        )
    if not distribution_file:
        return eval_data
    else:
        return eval_data, all_const


def show_text_pred_gold(llm_out, gold_data):
    """
    Side-by-side display of text input with gold and llm-predicted labels
    """
    # label mapping, all non-argumentative labels mapped to "nag" (nicht-argumentativ)
    map_to_6cl = {
        'gld': 'nag',
        'faz': 'nag',
        'ahg': 'nag',
        'whg': 'nag',
        'son': 'nag'
    }

    warn = "Unequal num of total essay samples! gold_data has {}, llm_out has {}".format(
        len(gold_data), len(llm_out)
    )
    assert len(gold_data) == len(llm_out), warn

    for i in range(len(gold_data)):
        essay_data, pred_labels = gold_data[i], llm_helper.parse_llm_out(llm_out[i])
        try:
            assert len(essay_data) == len(pred_labels)
        except AssertionError:
            print("Unequal num of labels in essay {}".format(i))
            print('_' * 50 + '\n')
            continue
        for idx in range(len(essay_data)):
            # map gold label
            gold_lab = map_gold_labels(essay_data[idx][1].lower(), map_to_6cl)
            # llm_output has already been mapped
            print('\t'.join([essay_data[idx][0], gold_lab, pred_labels[idx]]))
        print('_' * 50 + '\n')
    return


def show_essay_pred_by_id(llm_out, gold_data, id_to_check):
    warn = "Unequal num of total essay samples! gold_data has {}, llm_out has {}".format(
        len(gold_data), len(llm_out)
    )
    assert len(gold_data) == len(llm_out), warn

    print('Input essay:')
    essay = gold_data[id_to_check]
    for i in range(len(essay)):
        print(str(i + 1) + '\t', essay[i])
    print()
    print('Output:')
    print(llm_out[id_to_check])
    return


def full_eval(llm_out, gold_data, gold_consts):
    """
    Classification evaluation of LLM output, llm_out and gold_data must be aready read-in from files into lists
    """

    # label mapping, all non-argumentative labels mapped to "nag" (nicht-argumentativ)
    map_to_6cl = {
        'gld': 'nag',
        'faz': 'nag',
        'ahg': 'nag',
        'whg': 'nag',
        'son': 'nag'
    }
    # label mapping, same as above but leaving FAZ unmapped.
    map_to_7cl = {
        'gld': 'nag',
        'ahg': 'nag',
        'whg': 'nag',
        'son': 'nag'
    }

    warn = "Unequal num of total essay samples! gold_data has {}, llm_out has {}, gold_consts has {}".format(
        len(gold_data), len(llm_out), len(gold_consts)
    )
    assert len(gold_data) == len(llm_out) == len(gold_consts), warn

    model_preds = []
    pred_consts = []
    golds_w_faz = []
    golds_no_faz = []
    num_bad_output = 0
    for i in range(len(gold_data)):
        # Eval of constellation
        pred_const = llm_helper.parse_llm_out_for_const(llm_out[i])
        pred_consts.append(pred_const)

        essay_data = gold_data[i]
        try:
            pred_labels = llm_helper.parse_llm_out(llm_out[i])
        except KeyError:
            # if outputted label could not be mapped,
            # skip this essay, save bad output, continue to next essay
            print('Bad key found')
            pred_labels = []

        # if unequal num of labels in essay, i.e. llm-output is bad, skip
        try:
            assert len(essay_data) == len(pred_labels)
        except AssertionError:
            print("Unequal num of labels in essay {}".format(i))
            print(f"Gold data has {len(essay_data)}, predicted has {len(pred_labels)} labels")
            num_bad_output += 1
            continue

        # map gold labels, llm_out will have been mapped by helper script already
        # w_faz means not mapping "faz" so it can be extracted and excluded
        gold_labels_w_faz = [map_gold_labels(lab.lower(), map_to_7cl) for text, lab in essay_data]
        gold_labels_no_faz = [map_gold_labels(lab.lower(), map_to_6cl) for text, lab in essay_data]

        model_preds += pred_labels
        golds_w_faz += gold_labels_w_faz
        golds_no_faz += gold_labels_no_faz

    np.set_printoptions(suppress=True)
    assert len(golds_no_faz) == len(golds_w_faz) == len(model_preds)
    print('total no. gold labels', len(golds_no_faz))

    print('No. of bad outputs:', num_bad_output)

    # exclude FAZ from evaluation
    golds_minus_faz, preds_minus_faz = [], []
    for gold, pred in zip(golds_w_faz, model_preds):
        if gold != 'faz':
            golds_minus_faz.append(gold)
            preds_minus_faz.append(pred)

    # First, evaluation of prediction of overall constellation
    gold_consts_mod, pred_consts_mod = [], []
    for gold, pred in zip(gold_consts, pred_consts):
        if pred is not None:
            gold_consts_mod.append(gold)
            pred_consts_mod.append(pred)
    print('Gesamtkonstellation Eval based on {}/{}:'.format(len(gold_consts_mod), len(gold_consts)))
    acc, prfs = evaluate_classification(gold_consts_mod, pred_consts_mod, average=None)
    print('Accuracy:', round(acc, 3))
    print(prfs)

    print('-' * 30)
    print('Standard evaluation (FAZ INcluded in eval)')
    # standard classification evaluation
    acc, prfs = evaluate_classification(golds_no_faz, model_preds, average="macro")
    print('Accuracy:', acc)
    print('Macro average PRFS:')
    print(prfs)
    _, prfs = evaluate_classification(golds_no_faz, model_preds, average=None)
    print('Detailed PRFS:')
    print("con", "nag", "pro", "th1", "th2", "zth")
    print(np.round(prfs, decimals=3))
    # display conf mat
    labs = ["con", "nag", "pro", "th1", "th2", "zth"]
    print_confmat(golds=golds_no_faz, preds=model_preds, labels=labs)

    print('-'*30)
    print('Evaluation with exclusion of FAZ labels from evaluation')
    print(f'{len(golds_w_faz) - len(golds_minus_faz)} / {len(golds_w_faz)} excluded')
    # standard classification evaluation
    acc, prfs = evaluate_classification(golds_minus_faz, preds_minus_faz, average="macro")
    print('Accuracy:', acc)
    print('Macro average PRFS:')
    print(prfs)
    _, prfs = evaluate_classification(golds_minus_faz, preds_minus_faz, average=None)
    print('Detailed PRFS:')
    print("con", "nag", "pro", "th1", "th2", "zth")
    print(np.round(prfs, decimals=3))
    # display conf mat
    labs = ["con", "nag", "pro", "th1", "th2", "zth"]
    print_confmat(golds=golds_minus_faz, preds=preds_minus_faz, labels=labs)
    return


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


def evaluate_classification(Ygold, Ypred, average='macro'):
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


def print_confmat(golds, preds, labels=None):
    """ Display confusion matrix """
    # rename some labels for paper
    mapper = {'zth': 'cth', 'nag': 'n-a'}
    golds = [mapper[lab] if lab in mapper else lab for lab in golds]
    preds = [mapper[lab] if lab in mapper else lab for lab in preds]

    labels = ("cth", "th1", "th2", "pro", "con", "n-a")
    cm = confusion_matrix(y_true=golds, y_pred=preds, normalize='pred', labels=labels)
    # cm = confusion_matrix(y_true=golds, y_pred=preds, labels=labels)
    cm = np.round(cm, decimals=2)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    # plt.show()
    return


def get_confmat_llm(setting_name_base, plot_title=None, output_file_name=None, num_folds=5):
    """
    setting_name_base: something like "llama70_k4_"
    """
    # Show normalised confusion matrix, averaged across folds

    # map "zth" and "nag" to label terms used in paper
    mapper = {'zth': 'cth', 'nag': 'n-a'}

    # desired order of labels
    ordered_labs = ["cth", "th1", "th2", "pro", "con", "n-a"]

    cms = []
    for i in range(num_folds):
        data = pickle.load(open('{}fold{}.p'.format(setting_name_base, i + 1), 'rb'))
        ypreds = data[0]
        ygolds = data[1]
        assert len(ypreds) == len(ygolds)

        # map to terms used in paper
        ygolds = [mapper[lab] if lab in mapper else lab for lab in ygolds]
        ypreds = [mapper[lab] if lab in mapper else lab for lab in ypreds]

        cm = confusion_matrix(y_true=ygolds, y_pred=ypreds, normalize='true', labels=ordered_labs)
        cms.append(cm)

    allcms = np.array(cms)
    norm_cm = np.around(np.mean(allcms, axis=0), decimals=2)
    print(norm_cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=norm_cm, display_labels=ordered_labs)

    disp.plot()
    if plot_title is not None:
        plt.title(plot_title)
    # plt.show()
    if output_file_name is not None:
        plt.savefig(output_file_name, transparent=False)
        print('Image saved to', output_file_name)
    # print(test_labels)


def get_prfs_llm(setting_name_base, num_folds=5):
    """
    setting_name_base: something like "llama70_k4_"
    """
    # map "zth" and "nag" to label terms used in paper
    mapper = {'zth': 'cth', 'nag': 'n-a'}

    # desired order of labels
    ordered_labs = ["cth", "th1", "th2", "pro", "con", "n-a"]

    prfs_s_by_class = []
    for i in range(num_folds):
        data = pickle.load(open('{}fold{}.p'.format(setting_name_base, i + 1), 'rb'))
        ypreds = data[0]
        ygolds = data[1]
        assert len(ypreds) == len(ygolds)

        # map to terms used in paper
        ygolds = [mapper[lab] if lab in mapper else lab for lab in ygolds]
        ypreds = [mapper[lab] if lab in mapper else lab for lab in ypreds]

        prfs_by_class = precision_recall_fscore_support(
            y_true=ygolds,
            y_pred=ypreds,
            average=None,
            labels=ordered_labs
        )
        prfs_s_by_class.append(prfs_by_class)

    print(f'PRF by class across {num_folds} folds: ')
    print(llm_eval.print_avg_prfs_by_class(prfs_s_by_class, num_folds=num_folds, num_classes=6))
