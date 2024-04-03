import glob
import logging
import os
import random
from argparse import ArgumentParser
from typing import List, Dict, Tuple

from nelbio.utils.io import copy_concept_files


def group_data_by_cui_and_str(concept_list: List[List[str]]) -> \
        Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    cui2data: Dict[str, List[str]] = {}
    mention2data: Dict[str, List[str]] = {}
    mention_cui2data: Dict[str, List[str]] = {}
    for concept_attrs in concept_list:
        cui = concept_attrs[-1].strip()
        mention = concept_attrs[-2].strip()
        mention_cui = f"{mention}||{cui}"

        if cui2data.get(cui) is None:
            cui2data[cui] = []
        if mention2data.get(mention) is None:
            mention2data[mention] = []
        if mention_cui2data.get(mention_cui) is None:
            mention_cui2data[mention_cui] = []
        cui2data[cui] = concept_attrs
        mention2data[mention] = concept_attrs
        mention_cui2data[mention_cui] = concept_attrs
    return cui2data, mention2data, mention_cui2data


def main(args):
    random.seed(32)
    output_dir = args.output_dir
    output_train_dir = os.path.join(output_dir, "train/")
    output_dev_dir = os.path.join(output_dir, "dev/")
    output_test_dir = os.path.join(output_dir, "test/")

    if not os.path.exists(output_train_dir):
        os.makedirs(output_train_dir)
    if not os.path.exists(output_dev_dir):
        os.makedirs(output_dev_dir)
    if not os.path.exists(output_test_dir):
        os.makedirs(output_test_dir)

    all_concept_files = []
    for data_split_name in ("train", "dev", "test"):
        data_split_dir = os.path.join(args.data_dir, f"{data_split_name}/")
        data_split_concept_files = glob.glob(os.path.join(data_split_dir, "*.concept"))
        all_concept_files.extend(data_split_concept_files)
    num_concept_files = len(all_concept_files)
    logging.info(f"There are {num_concept_files} concept files")

    random.shuffle(all_concept_files)
    num_train_dev_concept_files = int(num_concept_files * args.train_dev_size)
    num_dev_concept_files = int(num_train_dev_concept_files * args.dev_size)

    train_dev_concept_files = all_concept_files[:num_train_dev_concept_files]
    test_concept_files = all_concept_files[num_train_dev_concept_files:]
    train_concept_files = train_dev_concept_files[num_dev_concept_files:]
    dev_concept_files = train_dev_concept_files[:num_dev_concept_files]

    logging.info(f"Finished splitting data. Train: {len(train_concept_files)}, Dev: {len(dev_concept_files)}, "
                 f"Test: {len(test_concept_files)}")
    copy_concept_files(concept_file_paths=train_concept_files, out_dir=output_train_dir)
    copy_concept_files(concept_file_paths=dev_concept_files, out_dir=output_dev_dir)
    copy_concept_files(concept_file_paths=test_concept_files, out_dir=output_test_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--train_dev_size', type=float)
    parser.add_argument('--dev_size', type=float)
    parser.add_argument('--output_dir')
    args = parser.parse_args()
    main(args)
