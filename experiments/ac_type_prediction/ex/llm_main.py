"""
Classification experiments using LLMs
Access to model through Groq API
Zero_shot, no example essays
"""

import time
import os
import random
import pickle
from dotenv import load_dotenv
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from project_core import llm_helper
from project_core import llms
from project_core.prompt_templates import all_prompt_templates
from project_core.process_data import process_data_dir, prep_function_cls


# Load env variables (.env)
load_dotenv()

# API related
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

############# MODELS ################
# Possible model names
# LLM_MODEL_NAME = "llama-3.1-8b-instant"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
######################################
LLM_OUT_DIR = "llm_output_all"
LLM_OUT_NAME = "llama70_zero_steps"

NUM_ESSAYS_UNIT = 10

######################################
print('Experiment params:')
print('LLM model used:', LLM_MODEL_NAME)
print()


def process_essays(list_essays, lower=True):
    """
    Process list of labelled essays
    :param list_essays: list(sentence, cz_tag))
    :param lower: bool, whether or not to lower-case labels
    :return essay_texts: list of sentence-segmented essay texts where each text is a single,
    multi-line string of the form [sentence_id]:[sentence_text]\n
    labels: index-aligned list of labels of the form list(list(labels))
    """
    essay_texts, labels = [], []
    for essay in list_essays:
        target_essay = ""
        labs = []
        for i, sent_lab in enumerate(essay):
            sent, lab = sent_lab
            if lower:
                labs.append(lab.lower())
            else:
                labs.append()
            target_sent = str(i + 1) + ': ' + sent + '\n'
            target_essay += target_sent

        essay_texts.append(target_essay)
        labels.append(labs)

    return essay_texts, labels


def gen_prompt_zero(target_essay_processed):
    """
    Generate prompt to feed to LLM in a zero-shot prompt scenario
    :param target_essay_processed: essay text of the target essay as a single multi-line string
    """
    # System prompt
    system_prompt = all_prompt_templates["system_prompt"]
    # optionally add extended
    # system_prompt += all_prompt_templates["system_prompt_extended"]

    # User prompt
    user_p_background = ''.join([
        all_prompt_templates["user_prompt_task_general"],
        all_prompt_templates["user_prompt_overall_constellation"],
        all_prompt_templates["user_prompt_zones_with_oc"],
        # all_prompt_templates["user_prompt_concl_additional"],
        all_prompt_templates["user_prompt_anno_steps"]
    ])
    user_p_command = all_prompt_templates["user_prompt_command_with_oc"].format(target_essay_processed)
    user_prompt = user_p_background + user_p_command
    return system_prompt, user_prompt


def gen_prompt_few_shot(target_essay_processed, demo_essays_processed):
    """
    Generate prompt to feed to LLM in few-shot scenario
    Args:
        target_essay_processed: str essay text of the target essay as a single multi-line string
        demo_essays_processed: list(tuple(demo essay, demo output)), list of processed demo examples where each example
        is a tuple of demo essay (str) and demo output (str)
    Return:
        system_prompt:str System prompt for the whole chat turn, including the task background
        user_p_command_target:str User prompt including the TARGET essay
        user_prompt_demos:list(str) User prompts with each DEMO essay, index-aligned with "demos_out"
        demos_out:list(str) demo outputs, index-aligned with user_prompt_demos
    """
    # System prompt
    system_prompt = all_prompt_templates["system_prompt"]
    # optionally add extended
    # system_prompt += all_prompt_templates["system_prompt_extended"]

    # User prompt
    user_p_background = ''.join([
        all_prompt_templates["user_prompt_task_general"],
        all_prompt_templates["user_prompt_overall_constellation"],
        all_prompt_templates["user_prompt_zones_with_oc"],
        # all_prompt_templates["user_prompt_concl_additional"],
        # all_prompt_templates["user_prompt_anno_steps"]
    ])
    user_p_command_demos, demos_out = [], []
    for demo_essay, demo_out in demo_essays_processed:
        user_command_demo = all_prompt_templates["user_prompt_command_with_oc"].format(demo_essay)
        user_command_demo = user_p_background + user_command_demo
        user_p_command_demos.append(user_command_demo)
        demos_out.append(demo_out)
    user_p_command_target = all_prompt_templates["user_prompt_command_with_oc"].format(target_essay_processed)
    user_p_command_target = user_p_background + user_p_command_target

    return system_prompt, user_p_command_target, user_p_command_demos, demos_out


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


def run_experiment(data_dir, ex_outdir):
    # Set up data to use
    print(f'Fetching data from {data_dir}')
    data_orig = process_data_dir(data_dir)
    data_orig = prep_function_cls(data_orig)

    # Label set to use, anything not in this set to be mapped to "SON"
    valid_labels = ("ZTH", "TH1", "TH2", "PRO", "CON", "SON")
    # Map data to use valid labels only
    data = []
    for essay in data_orig:
        essay_mapped = []
        for seg_data in essay:
            # seg_data is [seg_text, lab, topic, const], with topic and const being optional
            # seg_data[1] is the label
            lab = seg_data[1]
            if lab not in valid_labels:
                seg_data[1] = "son"
            else:
                seg_data[1] = lab.lower()
            essay_mapped.append(seg_data)
        data.append(essay_mapped)

    print(f'Total: {len(data)} essays')

    # set up LLM model
    llm_model = llms.GroqClassifier(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL_NAME
    )

    # process (test) data
    essay_texts, labels = process_essays(data)

    # iteratively prompt LLM with each essay from (test) data:
    llm_raw_outputs, llm_bad_outputs = [], []
    model_preds, golds = [], []
    num_bad_output = 0
    process_count = 1
    print('Start analysing...')
    for idx in range(len(essay_texts)):
        if process_count % NUM_ESSAYS_UNIT == 0:
            print('Finished prompting {} essays'.format(process_count))
        essay = essay_texts[idx]
        gold_labs = labels[idx]

        # get prompt
        sys_prompt, usr_prompt = gen_prompt_zero(target_essay_processed=essay)
        # prompt LLM
        llm_outtext = llm_model.get_llm_completion_zero(
            system_prompt=sys_prompt,
            user_prompt=usr_prompt,
            temperature=1,
            max_out_tokens=None
        )

        # parse output
        llm_raw_outputs.append(llm_outtext)
        try:
            pred_essay_labs = llm_helper.parse_llm_out(llm_out=llm_outtext)
        except KeyError:
            # same action as above
            print('Bad output found')
            num_bad_output += 1
            llm_bad_outputs.append([essay, llm_outtext])
            process_count += 1
            continue

        if len(gold_labs) != len(pred_essay_labs):
            print('Bad output found (unequal label counts)')
            num_bad_output += 1
            llm_bad_outputs.append([essay, llm_outtext])
        else:
            model_preds += pred_essay_labs
            golds += gold_labs

        # increase counter
        process_count += 1

    # standard classification evaluation
    print(f'In total {num_bad_output} essays excluded due to bad output')
    acc, prfs = evaluate_classification(model_preds, golds, average="macro")
    print('Accuracy:', acc)
    print('PRFS:')
    print(prfs)

    # store LLM out
    # create the dir if it does not yet exist
    full_dir = ex_outdir + '/' + LLM_OUT_DIR
    print('Saving LLM output to dir:', full_dir, '+', LLM_OUT_NAME)
    Path(full_dir + '/raw').mkdir(parents=True, exist_ok=True)
    Path(full_dir + '/bad').mkdir(parents=True, exist_ok=True)

    fo = open(f'{full_dir}/raw/{LLM_OUT_NAME}.p', 'wb')
    pickle.dump(llm_raw_outputs, fo)
    fo.close()
    print('Created raw out file')

    fo = open(f'{full_dir}/bad/{LLM_OUT_NAME}.p', 'wb')
    pickle.dump(llm_bad_outputs, fo)
    fo.close()
    print('Created bad out file')


if __name__ == '__main__':
    # input paths to test script
    datadir = ''
    ex_outdir = '.'
    run_experiment(data_dir=datadir, ex_outdir=ex_outdir)
