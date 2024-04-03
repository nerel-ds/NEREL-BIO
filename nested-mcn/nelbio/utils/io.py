import codecs
import glob
import logging
import os
from typing import Dict, List

import pandas as pd
import numpy as np
from tqdm import tqdm


def save_dict(d: Dict, save_path: str, sep='\t'):
    with codecs.open(save_path, 'w+', encoding="utf-8") as out_file:
        for k, v in d.items():
            assert sep not in str(k)
            assert sep not in str(v)
            out_file.write(f"{k}{sep}{v}\n")


def load_dict(inp_path: str, sep='\t') -> Dict[str, str]:
    d: Dict[str, str] = {}
    with codecs.open(inp_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            attrs = line.strip().split(sep)
            key = attrs[0]
            value = attrs[1]
            d[key] = value

    return d


def load_dictionary_tuples(inp_path: str) -> np.array:
    data = []
    logging.info(f"Loading vocab...")
    with open(inp_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, miniters=len(lines) // 100):
            line = line.strip()
            if line == "": continue
            cui, name = line.split("||")
            data.append((name, cui))

    logging.info(f"Vocab is loaded. There are {len(data)} <name, cui> pairs")
    data = np.array(data)
    return data


def read_mrconso(fpath) -> pd.DataFrame:
    columns = ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE',
               'STR', 'SRL', 'SUPPRESS', 'CVF', 'NOCOL']
    return pd.read_csv(fpath, names=columns, sep='|', encoding='utf-8', quoting=3)


def read_mrsty(fpath) -> pd.DataFrame:
    columns = ['CUI', 'TUI', 'STN', 'STY', 'ATUI', 'CVF', 'NOCOL']
    return pd.read_csv(fpath, names=columns, sep='|', encoding='utf-8', quoting=3)


def read_mrrel(fpath) -> pd.DataFrame:
    columns = ["CUI1", "AUI1", "STYPE1", "REL", "CUI2", "AUI2", "STYPE2", "RELA", "RUI", "SRUI", "RSAB", "VSAB",
               "SL", "RG", "DIR", "SUPPRESS", "CVF", 'NOCOL']
    return pd.read_csv(fpath, names=columns, sep='|', encoding='utf-8')


def read_mrdef(fpath) -> pd.DataFrame:
    columns = ["CUI", "AUI", "ATUI", "SATUI", "SAB", "DEF", "SUPPRESS", "CVF", 'NOCOL']
    return pd.read_csv(fpath, names=columns, sep='|', encoding='utf-8')


def read_sem_groups(fpath) -> pd.DataFrame:
    columns = ["Semantic Group Abbrev", "Semantic Group Name", "TUI", "Full Semantic Type Name"]
    return pd.read_csv(fpath, names=columns, sep='|', encoding='utf-8')


def load_biosyn_all_concepts(data_dir: str) -> List[List[str]]:
    data_samples = []
    for concept_fname in os.listdir(data_dir):
        concept_path = os.path.join(data_dir, concept_fname)
        with open(concept_path, 'r', encoding='utf-8') as inp_file:
            concept_data = [line.split('||') for line in inp_file]
            data_samples.extend(concept_data)
    return data_samples


def load_biosyn_data_groupby_concept(data_dir: str) -> Dict[str, List[List[str]]]:
    concept_id2data_samples: Dict[str, List[List[str]]] = {}
    concept_files = glob.glob(os.path.join(data_dir, "*.concept"))
    for concept_path in concept_files:
        concept_id = concept_path.split('/')[-1].split('.')[0]
        assert concept_id2data_samples.get(concept_id) is None

        with open(concept_path, 'r', encoding='utf-8') as inp_file:
            concept_data = [line.split('||') for line in inp_file]
            concept_id2data_samples[concept_id] = concept_data
    return concept_id2data_samples


def copy_concept_files(concept_file_paths: List[str], out_dir: str):
    for concept_path in concept_file_paths:
        concept_name = concept_path.split('/')[-1]
        output_concept_path = os.path.join(out_dir, concept_name)
        with codecs.open(concept_path, 'r', encoding="utf-8") as in_file, \
                codecs.open(output_concept_path, 'w+', encoding="utf-8") as out_file:
            out_file.write(in_file.read())


def load_biosyn_formated_sep_context_dataset(input_dir, cui_dictionary=None, drop_duplicates=False, drop_cuiless=False,
                                             drop_not_nested=False):
    sep_contexts = []
    flat_cuis = []
    flat_queries = []
    n_m_list = []
    counter = 0
    seen_samples = None
    if drop_duplicates:
        seen_samples = set()
    cuiless_counter = 0
    not_nested_counter = 0
    for fname in os.listdir(input_dir):
        if not fname.endswith("concept"):
            continue
        fpath = os.path.join(input_dir, fname)
        with open(fpath, 'r', encoding="utf-8") as inp_file:
            for line in inp_file:
                counter += 1
                attrs = line.strip().split('||')
                flat_cui = attrs[-1]
                if drop_cuiless and flat_cui not in cui_dictionary:
                    cuiless_counter += 1
                    continue
                flat_q = attrs[-2]
                sep_context = attrs[-3]
                nested_mentions = attrs[0].split('|')
                if drop_not_nested and len(nested_mentions) == 1:
                    not_nested_counter += 1
                    continue
                if drop_duplicates:
                    s = f"{flat_q}||{flat_cui}||{sep_context}"
                    if s in seen_samples:
                        continue
                    seen_samples.add(s)

                flat_cuis.append(flat_cui)
                flat_queries.append(flat_q)
                sep_contexts.append(sep_context)
                n_m_list.append(nested_mentions)
    num_duplicates = counter - len(flat_queries)
    if seen_samples is not None:
        assert len(flat_queries) == len(seen_samples)
    logging.info(f"Loaded dataset: {len(flat_queries)} samples. Dropped {num_duplicates} duplicates and cuiless "
                 f"({cuiless_counter} CUI-less and {not_nested_counter} not nested entities)")

    return sep_contexts, flat_cuis, flat_queries, n_m_list


def save_sep_context_dataset(output_dir, sep_contexts, flat_cuis, flat_queries, n_m_list):
    output_path = os.path.join(output_dir, "0.concept")
    with open(output_path, 'w', encoding="utf-8") as out_file:
        for sp, fc, fq, nm_lst in zip(sep_contexts, flat_cuis, flat_queries, n_m_list):
            out_file.write(f"{'|'.join(nm_lst)}||-1|-1||{sp}||{fq}||{fc}\n")


