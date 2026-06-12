"""
Helper functions to support LLM-based classification
"""
import pickle
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os
import xml.etree.ElementTree as ET


def essay_data_from_grapat(xml_essay_file):
    """
    Prepping the annotated essay by parsing the Grapat output file (XML) to labelled sentence - label pairs
    """
    root = ET.parse(xml_essay_file).getroot()
    essay_data = []
    # looping through children nodes
    for child in root:
        # currently only interested in content zone classification, hence only children with tag "edu"
        if child.tag == 'edu':
            text = child.text # access cdata text
            match = re.search(r'\[(.*)\]([A-Z]+)', text)
            try:
                sent, lab = match.group(1), match.group(2)
                essay_data.append((sent, lab))
            except IndexError:
                print('Problem with sentence:', text)
    return essay_data


def essay_data_from_txt(txt_essay_file):
    """
    Prepping the annotated essay by extracting labelled sentence - label pairs from *.txt files
    """
    essay_data = []
    with open(txt_essay_file, 'r') as fi:
        for line in fi:
            # some lines might be blank or near-blank lines, skip those
            if len(line.strip()) > 1:
                match = re.search(r'[\[<](.*)[\]>]([A-Z12]+)', line)
                if not match:
                    print(f'No match found for file f{txt_essay_file}, line f{line}')
                else:
                    try:
                        sent, lab = match.group(1), match.group(2)
                        essay_data.append((sent, lab))
                    except IndexError:
                        print('Problem with sentence:', line)
    return essay_data


def get_data_all_essays(dir_to_essay_files, file_format='txt', demo_essays=None):
    """
    :param dir_to_essay_files: Directory in which all annotated essay data files live (as xml or txt)
    :param file_format: Extension of each essay data file, 'txt' or 'xml'
    :param demo_essays: dict, Demo essays as mappings "ESSAY_ID": "GESAMTKONSTELLATION"
    :return: list(list((sent, lab))) -> labelled test essays, list(list((sent, lab))) -> labelled demo essays
    """
    all_data, demo_data = [], []
    if file_format not in ('txt', 'xml'):
        raise ValueError('Invalid essay file format. Options are "xml" and "txt"')
    for fname in os.listdir(dir_to_essay_files):
        if fname.endswith(file_format):
            eid = str(fname)[:-4]
            if file_format == 'txt':
                essay_data = essay_data_from_txt(os.path.join(dir_to_essay_files, fname))
            else:
                essay_data = essay_data_from_grapat(os.path.join(dir_to_essay_files, fname))
            if demo_essays:
                if eid in demo_essays:
                    # append demo essay data paired with gesamtkonstellation
                    demo_data.append((essay_data, demo_essays[eid]))
                else:
                    all_data.append(essay_data)
            else:
                all_data.append(essay_data)
    return all_data, demo_data


def get_all_essay_const_topic(dir_to_essay_files, distribution_file, demo_essays=None):
    """
    Extract all overall constellations in the training data
    :param dir_to_essay_files: Directory in which all annotated essay data files live (as xml or txt)
    :param distribution_file: TSV file
    :param demo_essays: dict, Demo essays as mappings "ESSAY_ID": "GESAMTKONSTELLATION"
    :return:
    """
    eid2const, eid2topic = read_essay_distribution_file(distribution_file, map_const_to_2=True)
    all_const, all_topics = [], []
    for fname in os.listdir(dir_to_essay_files):
        if fname.endswith('txt'):
            eid = str(fname)[:-4]
            e_const = eid2const[eid]
            e_topic = eid2topic[eid]
            # store this constellation and topic unless present essay is a demo essay
            if not demo_essays:
                all_const.append(e_const)
                all_topics.append(e_topic)
            elif eid not in demo_essays:
                all_const.append(e_const)
                all_topics.append(e_topic)
    return all_const, all_topics


def read_essay_distribution_file(target_file, map_const_to_2=True):
    """
    Read in the file containing essay-specific info on writing topic and argumentative constellation
    :param target_file: TSV file
    :param map_const_to_2: bool, map the 4 overall constellation labels to 2
    :return: eid2const dict(): mapping from essay id to overall constellation label
            eid2topic dict(): mapping from essay id to topic label
    """
    const_mapping = {
        "Deliberative": "Entschieden",
        "Undecided": "Unentschieden",
        "One-sided": "Entschieden",
        "Unclear": "Unentschieden"
    }
    eid2const, eid2topic = dict(), dict()
    with open(target_file, 'r') as fi:
        # skip header row
        next(fi)
        for line in fi:
            try:
                eid, topic, const = line.strip().split('\t')
            except ValueError as e:
                print(e)
                print(line.strip())
                continue
            assert eid.isdigit(), print(f'Error with eid {eid}')
            if not map_const_to_2:
                eid2const[eid] = const
            else:
                eid2const[eid] = const_mapping[const]
            eid2topic[eid] = topic
    return eid2const, eid2topic


def get_essays_avg_len(dir_to_essay_files):
    """
    Essays average length
    """
    all_data, _ = get_data_all_essays(dir_to_essay_files=dir_to_essay_files)
    essays_len = 0
    for essay in all_data:
        essay_len = 0
        for sent, _ in essay:
            essay_len += len(sent.split())
        essays_len += essay_len
    return essays_len / len(all_data)


def visualise_label_distribution(dir_to_essay_files):
    """
    Label dist
    """
    all_data, _ = get_data_all_essays(dir_to_essay_files=dir_to_essay_files)
    print(f'Total of {len(all_data)} essays.')
    map_to_6cl = {
        'gld': 'nag',
        'faz': 'nag',
        'ahg': 'nag',
        'whg': 'nag',
        'son': 'nag'
    }
    labs = []
    for essay in all_data:
        for sent, label in essay:
            lab = map_to_6cl[label.lower()] if label.lower() in map_to_6cl else label.lower()
            # lab = label.lower()
            labs.append(lab)

    print(f'Total of {len(labs)} labels')
    counter = Counter(labs)
    label_counts = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    # labels = ['info_intro', 'article_pro', 'article_con', 'own', 'other']
    # counts = [counter[x] for x in labels]
    print(label_counts)
    labels = [lab for (lab, freq) in label_counts]
    counts = [freq for (lab, freq) in label_counts]

    # for plot generation use the final label names used in paper
    map_to_paperlabs = {'nag': 'n-a', 'zth': 'cth'}
    labels = [map_to_paperlabs[lab] if lab in map_to_paperlabs else lab for lab in labels]

    print(labels)

    fig = plt.figure()
    plt.bar(labels, counts, width=0.8, color='darkblue', edgecolor='black')
    plt.xticks(rotation=20, ha="right")
    # plt.tick_params(axis='x', labelsize=8)

    # display count numbers on top of each bar
    for i in range(len(counts)):
        plt.text(i, counts[i] + 3, s=counts[i], ha='center')

    plt.xlabel('AC type labels')
    plt.ylabel('Occurrence frequency')
    plt.title('Distribution of AC Type Labels')

    # y-values on top of bar chart
    # for idx, val in enumerate(counts):
    #    plt.text(labels[idx] - 0.15, val + max(counts)*0.01, str(val))
    plt.show()
    # fig.savefig('zone_dist_50essays.png')
    return


def process_demo_essay(demo_essay_data):
    """
       Process a single essay to be used as demonstration in one-shot prompting scenario
       Args:
           demo_essay_data: tup(list(sentence, label), str):
           Tuple consisting of essay as list(sentence, label), matching Gesamtkonstellation
       Returns:
           demo_essay: str Essay as a single string to be incorporated into prompt
           demo_out: str Desired LLM out values as a single string to be incorporated into prompt
    """
    # process example sentences for one-shot-learning
    # labs2llm = {
    #     "ZTH": "Zentrale_These",
    #     "TH1": "These_1",
    #     "TH2": "These_2",
    #     "PRO": "Pro_Argument",
    #     "CON": "Con_Argument",
    #     "GLD": "Sonstiges",
    #     "AHG": "Sonstiges",
    #     "WHG": "Sonstiges",
    #     "SON": "Sonstiges",
    # }

    labs2llm = {
        "zth": "Zentrale_These",
        "th1": "These_1",
        "th2": "These_2",
        "pro": "Pro_Argument",
        "con": "Con_Argument",
        "son": "Sonstiges",
    }

    demo_essay = ""
    demo_sent_lab = demo_essay_data[0]
    # Prefix desired demo_out with the "gold" Gesamtkonstellation
    demo_out = demo_essay_data[1] + '\n'
    for idx in range(len(demo_sent_lab)):
        demo_sent = str(idx + 1) + ': ' + demo_sent_lab[idx][0] + '\n'
        gold_lab = labs2llm[demo_sent_lab[idx][1]]
        demo_lab = str(idx + 1) + ': ' + gold_lab + '\n'
        demo_essay += demo_sent
        demo_out += demo_lab
    return demo_essay, demo_out


def parse_llm_out(llm_out):
    """
    Extract content zone labels from the LLM output text
    :param llm_out:str LLM output text for a given essay
    :return: list() extracted list of labels
    """
    llm2labs = {
        "sonstiges": "son",
        "pro_argument": "pro",
        "con_argument": "con",
        "these_1": "th1",
        "these_2": "th2",
        "zentrale_these": "zth"
    }
    # check if it contains chunks of "thinking" (e.g. as done by qwen3_32b)
    if '</think>' in llm_out:
        llm_out = llm_out.split('</think>')[1]

    verdicts = llm_out.split('\n')
    # use regex to search for label match in LLM output
    rpattern = r'\d{1,2}:\s?([a-zA-Z_12]+)'
    labs = []
    for v in verdicts:
        # disregard possible LLM output lines that do NOT contain label predictions
        # TODO Add statement for extracting const
        # this also includes the verdict on "Gesamtkonstellation" which we are not extracting for eval
        if not re.search(rpattern, v):
            continue
        else:
            lab = re.search(rpattern, v).group(1)
            try:
                lab = llm2labs[lab.lower()]
            except KeyError:
                print('Invalid label in LLM output:', lab)
                raise KeyError
            labs.append(lab)
    return labs


def parse_llm_out_for_const(llm_out):
    """
    Extract the constellation label from llm_out
    :param llm_out:str LLM output text for a given essay
    :return: str the extracted const label
    """
    # check if it contains chunks of "thinking" (e.g. as done by qwen3_32b)
    if '</think>' in llm_out:
        llm_out = llm_out.split('</think>')[1]
    verdicts = llm_out.split('\n')
    rpattern = r'^(Entschieden|Unentschieden)'
    for v in verdicts:
        if re.search(rpattern, v):
            pred_const = re.search(rpattern, v).group(1)
            break
    else:
        print('No Gesamtkonstellation found, setting to "None".')
        pred_const = None
    return pred_const
