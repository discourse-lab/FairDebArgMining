"""
Classification experiments using LLMs
Access to model through Groq API (Llama and GPT-OSS), OpenAI API (others)
"""
import statistics
import os
import pickle
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sentence_transformers import SentenceTransformer

from project_core import llm_helper
from project_core import llms
from project_core.llm_few_shot import get_similar_demo_essays, get_random_demo_essays
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
SBERT_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
######################################

# Experiment Params
NUM_FOLDS = 3
# choosing top K ranked demo essays for ICL / k-shot prompting
K = 10
# "random" for randomly selected examples for ICL, "similarity" for similarity-based selection
# DEMO_SELECTION = "random"
DEMO_SELECTION = "similarity"

if DEMO_SELECTION == "similarity":
    LLM_OUT_DIR = "llm_output_all/out_similarity_fewshot"
else:
    LLM_OUT_DIR = "llm_output_all/out_random_fewshot"

LLM_OUT_NAME = f"llama70_k{K}"
# LLM_OUT_NAME = f"llama8_k{K}"

NUM_ESSAYS_UNIT = 10

######################################
print('Experiment params:')
print('LLM model used:', LLM_MODEL_NAME)
print('SBERT model used (if applicable):', SBERT_MODEL_NAME)
print(f'k-shot prompting, k = {K}, selection = {DEMO_SELECTION}')
print('Saving LLM output to dir:', LLM_OUT_DIR, '+', LLM_OUT_NAME)
print()

np.random.RandomState(2025)


def process_essay(essay_data, lower=True):
    """
    Process a single labelled essay
    :param essay_data:list(tuple(sentence, cz_tag, topic_id))
    s. also process_essays
    """
    essay_text = ""
    labels = []
    constellation = ""
    topic = ""

    for i, sent_lab_const_topic in enumerate(essay_data):
        sent, lab, const, t = sent_lab_const_topic
        if lower:
            labels.append(lab.lower())
        else:
            labels.append(lab)
        target_sent = str(i + 1) + ': ' + sent + '\n'
        essay_text += target_sent
        constellation = const
        topic = t
    return essay_text, labels, constellation, topic


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
        all_prompt_templates["user_prompt_zones_with_oc"]
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


def run_experiment(data_dir, ex_outdir):

    # Set up data to use
    print(f'Fetching data from {data_dir}')
    data_orig = process_data_dir(data_dir)
    data_orig = prep_function_cls(data_orig, include_topic=True, include_const=True)

    # Label set to use, anything not in this set to be mapped to "SON"
    valid_labels = ("ZTH", "TH1", "TH2", "PRO", "CON", "SON")
    # Map data to use valid labels only
    data = []
    for essay in data_orig:
        essay_mapped = []
        for seg_data in essay:
            # seg_data is [seg_text, lab, const, topic], with topic and const being optional
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

    if DEMO_SELECTION == "similarity":
        # set up SBert model
        print('Setting up SBert model:', SBERT_MODEL_NAME)
        sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

    # set up cross-val with KFold
    data = np.array(data, dtype=object)
    np.random.shuffle(data)
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2026)

    # store eval by fold
    acc_s, prfs_s, prfs_s_byclass = [], [], []
    total_bad_output = 0

    print('-------- Starting analysis --------')
    current_fold = 0
    for train_i, test_i in kf.split(data):
        current_fold += 1
        if current_fold < 3:
            print('This is fold', current_fold)
            continue
        else:
            train = data[train_i].tolist()
            test = data[test_i].tolist()
            print(f'\nFold {current_fold}: train = {len(train)}, test = {len(test)}')

            # iteratively prompt LLM with each essay from test data:
            llm_raw_outputs, llm_bad_outputs = [], []
            model_preds, golds = [], []
            num_bad_output = 0
            process_count = 1
            for i in range(len(test)):
                # check/report progress
                if process_count % NUM_ESSAYS_UNIT == 0:
                    print('Finished prompting {} essays'.format(process_count))

                target_essay = test[i]
                t_essay_text, t_labels, t_const, t_topic = process_essay(target_essay)

                if DEMO_SELECTION == "similarity":
                    # fetch similarity-based demo essays for ICL / K-shot prompting
                    demo_essays = get_similar_demo_essays(
                        target_essay=target_essay,
                        all_train_essays=train,
                        sbert_model=sbert_model,
                        k=K
                    )
                elif DEMO_SELECTION == "random":
                    # fetch k randomly chosen demo essays
                    demo_essays = get_random_demo_essays(
                        target_essay=target_essay,
                        all_train_essays=train,
                        k=K
                    )
                else:
                    print('DEMO_SELECTION must be set to "similarity" or "random"!')
                    raise Exception

                # loop through all extracted demo essays
                demos_processed = []
                for demo in demo_essays:
                    demo_sent_lab = [(sent, lab) for sent, lab, _, _ in demo]
                    demo_const = demo[0][2]
                    d_essay, d_out = llm_helper.process_demo_essay(demo_essay_data=(demo_sent_lab, demo_const))
                    demos_processed.append((d_essay, d_out))

                # few-shot scenario
                # get prompt
                sys_prompt, usr_prompt_target, usr_prompt_demos, out_demos = gen_prompt_few_shot(
                    target_essay_processed=t_essay_text,
                    demo_essays_processed=demos_processed
                )

                # prompt LLM - few shot
                llm_outtext = llm_model.get_llm_completion_few_shot(
                    user_prompt_target=usr_prompt_target,
                    user_prompt_demos=usr_prompt_demos,
                    assistant_out_demos=out_demos,
                    system_prompt=sys_prompt,
                    temperature=1,
                    # max_out_tokens=None
                )

                # parse output
                llm_raw_outputs.append(llm_outtext)
                try:
                    pred_essay_labs = llm_helper.parse_llm_out(llm_out=llm_outtext)
                except KeyError:
                    # same action as above
                    print('Bad output found')
                    num_bad_output += 1
                    llm_bad_outputs.append([t_essay_text, llm_outtext])
                    process_count += 1
                    continue

                if len(t_labels) != len(pred_essay_labs):
                    print('Bad output found (unequal label counts)')
                    num_bad_output += 1
                    llm_bad_outputs.append([t_essay_text, llm_outtext])
                else:
                    model_preds += pred_essay_labs
                    golds += t_labels

                # increase counter
                process_count += 1

            # standard classification evaluation
            print(f'\nFold {current_fold}: In total {num_bad_output} essays excluded due to bad output')
            acc, prfs = evaluate_classification(model_preds, golds, average="macro")
            print('Accuracy:', acc)
            print('PRFS:')
            print(prfs)
            acc_s.append(acc)
            prfs_s.append(prfs)
            total_bad_output += num_bad_output

            # store LLM out
            full_dir = ex_outdir + '/' + LLM_OUT_DIR
            print('Saving LLM output to dir:', full_dir, '+', LLM_OUT_NAME)
            Path(full_dir + '/raw').mkdir(parents=True, exist_ok=True)
            Path(full_dir + '/bad').mkdir(parents=True, exist_ok=True)
            Path(full_dir + '/pred_gold').mkdir(parents=True, exist_ok=True)

            fo = open(f'{full_dir}/raw/{LLM_OUT_NAME}_fold{current_fold}.p', 'wb')
            pickle.dump(llm_raw_outputs, fo)
            fo.close()
            print(f'Created raw out file for fold {current_fold}')

            fo = open(f'{full_dir}/bad/{LLM_OUT_NAME}_fold{current_fold}.p', 'wb')
            pickle.dump(llm_bad_outputs, fo)
            fo.close()
            print(f'Created bad out file for fold {current_fold}')

            # Store aligned predictions and golds
            fo = open(f'{full_dir}/pred_gold/{LLM_OUT_NAME}_fold{current_fold}.p', 'wb')
            pickle.dump((model_preds, golds), fo)
            fo.close()
            print(f'Created pred_gold out file for fold {current_fold}')

    print('Eval across {} folds:'.format(NUM_FOLDS))
    print('Total num of bad output:', total_bad_output)
    print('Accuracy', round(statistics.mean(acc_s), 3))
    print('Precision, Recall, F1')
    print(print_avg_prfs(prfs_s))


if __name__ == '__main__':
    # input paths to test script
    datadir = ''
    ex_outdir = ''
    run_experiment(data_dir=datadir, ex_outdir=ex_outdir)
