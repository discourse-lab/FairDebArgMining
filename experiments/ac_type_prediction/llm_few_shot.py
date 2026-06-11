"""
Script with functions for doing similarity-based and random-based few shot prompting
"""

import pickle
import random
from sentence_transformers import SentenceTransformer
import statistics
import numpy as np


def get_similar_demo_essays(target_essay, all_train_essays, sbert_model, k=1):
    """
    Get k selected demo essay(s) for a given target essay, based on simil
    Args
        target_essay: list(sent, lab)
        all_train_essays: list(list(sent, lab))
        sbert_model: loaded SBERT transformer model
        k: number of demo essays in ICL, defaults to 1
    """
    # get same-topic candidate essays from all train
    target_topic = target_essay[0][3]
    indexed_candidate_essays = get_same_topic_essays(target_topic, all_train_essays)
    # indexed_candidate_essays = list(enumerate(all_train_essays))
    # if k > total num of candidate essays, just take all cand essays to be demos, no need to compute simil etc.
    if k >= len(indexed_candidate_essays):
        demo_essays = [essay[1] for essay in indexed_candidate_essays]
        print('[NOTE!:] k > number of same-topic essays, using all same-topic essays as demos ')
        return demo_essays

    ranked_cand_eids = rank_candidate_essays(
        target_essay=target_essay,
        indexed_candidate_essays=indexed_candidate_essays,
        sbert_model=sbert_model
    )

    # get mapping from essay id to essay data among candidate essays
    eid2essaydata = dict(indexed_candidate_essays)
    # demo essays are the top k among the similarity ranked candidate essays
    demo_eids = ranked_cand_eids[:k]
    demo_essays = [eid2essaydata[eid] for eid in demo_eids]
    return demo_essays


def get_random_demo_essays(target_essay, all_train_essays, k=1):
    """
    Get k selected demo essay(s) for a given target essay, randomly chosen
    Args
        target_essay: list(sent, lab, const, topic)
        all_train_essays: list(list(sent, lab, const, topic))
        k: number of demo essays in ICL, defaults to 1
    """
    # get candidate essays from all train
    target_topic = target_essay[0][3]
    indexed_candidate_essays = get_same_topic_essays(target_topic, all_train_essays)
    # indexed_candidate_essays = list(enumerate(all_train_essays))
    # if k > total num of candidate essays, just take all cand essays to be demos, no need to compute simil etc.
    if k >= len(indexed_candidate_essays):
        demos = indexed_candidate_essays
        print('[NOTE!:] k > number of same-topic essays, using all same-topic essays as demos ')
    else:
        # select random sample
        demos = random.sample(indexed_candidate_essays, k=k)
    demo_essays = [indexed_essay[1] for indexed_essay in demos]
    return demo_essays


def get_same_topic_essays(target_topic, all_train_essays):
    """
    For a given target topic id, get all training essays on the same topic
    :param target_topic:str, Target topic, 'E-Book', 'Fast Food', 'performing arts', 'Voluntary Work'
    :param all_train_essays:list, all possible training essays as list(list(tuple(sent, label, const, topic)))
    :return list(tuple(essay_id, essaydata)) where essay data is list(tuple(sent, label, const, topic))
    """
    candidate_essays = []
    for sample in all_train_essays:
        # topic is the same for the whole essay
        essay_topic = sample[0][3]
        if essay_topic == target_topic:
            candidate_essays.append(sample)

    # add ids to all candidate essays so they are retrievable
    return list(enumerate(candidate_essays))


def rank_candidate_essays(target_essay, indexed_candidate_essays, sbert_model):
    """
    Rank essays
    target essay is tuple(sent, label, const, topic)
    candidate essays is list(tuple(essay_id, essay data))
    All essays as tuple(sent, label, tid)
    """
    eids_with_simil = []
    # will then sort eids_with_simil based on simil score, greater to smaller
    target_e_sents = essay_tup_to_essay_sentlist(target_essay)
    for eid, cand_essay in indexed_candidate_essays:
        cand_e_sents = essay_tup_to_essay_sentlist(cand_essay)
        simil = get_essay_similarity(
            t_sents=target_e_sents,
            c_sents=cand_e_sents,
            sbert_model=sbert_model
        )
        eids_with_simil.append((eid, simil))

    # rank all candidate essays by similarity score
    ranked = sorted(eids_with_simil, key=lambda x: x[1], reverse=True)
    ranked_eids = [eid for eid, _ in ranked]
    return ranked_eids


def essay_tup_to_essay_sentlist(essay_tup):
    """ Get essay text as sent list from essay data """
    sent_list = [sent for sent, _, _, _ in essay_tup]
    return sent_list


def get_essay_similarity(t_sents, c_sents, sbert_model):
    """
    Essay simil score.
    both essays as sent list, i.e. list(str)
    return: float: similarity score between the two essays according to SBERT as a single scalar
    """
    t_chunks = split_chunks(t_sents)
    c_chunks = split_chunks(c_sents)
    chunk_simils = []
    for t_chunk, c_chunk in zip(t_chunks, c_chunks):
        t_str = ' '.join(t_chunk)
        c_str = ' '.join(c_chunk)
        simil_score = get_chunk_simil(
            target_essay_str=t_str,
            cand_essay_str=c_str,
            sbert_model=sbert_model
        )
        chunk_simils.append(simil_score)

    # there should be 3 chunk simil scores
    assert len(chunk_simils) == 3, print('No. of chunk similarities is not 3!')

    # get essay similarity as weighted average of chunk simil
    # # weight: 1, 2, 2
    # essay_simil_score = 0.2 * chunk_simils[0] + 0.4 * chunk_simils[1] + 0.4 * chunk_simils[2]
    # equal weight, just take arithmetic mean
    essay_simil_score = statistics.mean(chunk_simils)
    return essay_simil_score


def split_chunks(essay_sent_list):
    """
    Split essay into 3 chunks, respecting sentence boundaries
    """
    split1 = int(len(essay_sent_list) * 0.33)
    split2 = int(len(essay_sent_list) * 0.66)
    head, body, tail = essay_sent_list[:split1], essay_sent_list[split1:split2], essay_sent_list[split2:]
    return [head, body, tail]


def get_chunk_simil(target_essay_str, cand_essay_str, sbert_model):
    """
    Get cosine simil between target and can chunks of text using SBERT
    """
    t_embeds, cand_embeds = sbert_model.encode(target_essay_str), sbert_model.encode(cand_essay_str)
    simil_score = sbert_model.similarity(t_embeds, cand_embeds).numpy()
    return float(np.squeeze(simil_score))


if __name__ == '__main__':
    fi = open('../data_tmp/full_cz_data.p', 'rb')
    data = pickle.load(fi)
    fi.close()

    # SBERT_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

    random.shuffle(data)
    test = data[0]
    train = data[1:11]

    # print('Setting up SBert model:', SBERT_MODEL_NAME)
    # sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

    # demos = get_similar_demo_essays(
    #     target_essay=test,
    #     all_train_essays=train,
    #     sbert_model=sbert_model
    # )

    demos = get_random_demo_essays(
        target_essay=test,
        all_train_essays=train,
        k=3
    )
