import os
import pickle
import random
import re
from pathlib import Path
from collections import Counter

random.seed(42)


def remove_punctuation(sample, lower=True):
    """ remove punctuation and optionally lower case everything in a sample (default behaviour)"""
    punc = r"[\.\,\;\:\?\!\(\)\"\']"
    if lower:
        new = re.sub(punc, '', sample.lower())
    else:
        new = re.sub(punc, '', sample)
    return new


def map_label(label, mapping_dict):
    return mapping_dict[label]


def map_labels(data, mapping_dict, topic_id=False):
    """
    :param data: list(list(sentence, cz_tag))
    """
    mapped = []
    for text in data:
        text_mapped = []
        if not topic_id:
            for sent, label in text:
                newlab = mapping_dict[label] if label in mapping_dict else label
                text_mapped.append((sent, newlab))
        else:
            for sent, label, tid in text:
                newlab = mapping_dict[label] if label in mapping_dict else label
                text_mapped.append((sent, newlab, tid))
        mapped.append(text_mapped)
    return mapped


def convert_essays_x_y(data, lab2index=None, lowercase_x=False):
    """
    :param data: list(list(sentence, argtype_label))
    :param lab2index: dict item manually mapping labels to index numbers, optional
    """
    x, y = [], []
    for text in data:
        for sent, label in text:
            sentence = sent.lower() if lowercase_x else sent
            x.append(sentence)
            lab = lab2index[label] if lab2index is not None else label
            y.append(lab)
    assert len(x) == len(y)
    return x, y


def count_labels(y):
    """
    Get label distribution
    :param y: list() of labels
    """
    c = Counter(y)
    # sort by key (i.e. label)
    sort = sorted(c.items(), key=lambda x: x[0])
    return sort
