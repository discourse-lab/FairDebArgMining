import pandas as pd
import numpy as np
import os


def process_data_dir(path_to_dir, sep='|', encoding='iso-8859-1'):
    data = []
    files_discarded = 0
    total_files = 0
    for fname in os.listdir(path_to_dir):
        if fname.endswith('.csv'):
            total_files += 1
            essay_data = process_anno_csv(
                csv_file=os.path.join(path_to_dir, fname),
                sep=sep,
                encoding=encoding
            )
            if essay_data is not None:
                data.append(essay_data)
            else:
                # this was a 'bad' essay that will not remain in the dataset further
                files_discarded += 1
    print(f'{files_discarded} / {total_files} are bad and discarded from dataset. {total_files - files_discarded} kept')
    return data


def process_anno_csv(csv_file, sep='|', encoding='iso-8859-1'):
    """
    Designated format for each essay:
    [Seg-level information], topic, constl]
    Essay-level info is
    [(segment, layer 1 label, layer 2 label, added_yes/no)]
    """
    df = pd.read_csv(
        csv_file,
        sep=sep,
        encoding=encoding
    )

    # First extract essay-level info, topic + constellation
    topic = df['THE'][0]
    const = df['KON'][0]

    # if either topic or const turns out to be 'NaN', skip this CSV and display file name
    if pd.isna(topic) or pd.isna(const):
        # print(f'NaN found in file {csv_file}, skipping it')
        return None

    # drop first row of df and extract segment-level info
    df = df.drop(0)
    text = df['Text']
    layer_1 = df['AUF']
    layer_2 = df['FKT']
    added = df['ADD']

    seg_data = list(zip(text, layer_1, layer_2, added))
    return [seg_data, topic, const]


def prep_function_cls(full_dataset, lower_labs=False, include_const=False, include_topic=False):
    """
    Pre-process the full dataset for classification experiments on the functional layer (i.e. arg component types)
    :param full_dataset list() All essays in format [[Seg-level information], topic, constl] with seg-level info
    being [(segment, layer structure label, layer function label, added_yes/no]
    """
    const_mapping = {
        "EIN": "Entschieden",
        "ABW": "Entschieden",
        "UEN": "Unentschieden",
        "UKL": "Unentschieden"
    }
    data = []
    for essay in full_dataset:
        segs = essay[0]
        topic = essay[1]
        const = essay[2]
        essay_data = []
        for seg, anno_struct, anno_funct, added_y in segs:
            seg_data = [seg, anno_funct.lower()] if lower_labs else [seg, anno_funct]
            if include_const:
                seg_data.append(const_mapping[const])
            if include_topic:
                seg_data.append(topic)
            essay_data.append(seg_data)
        data.append(essay_data)
    return data
