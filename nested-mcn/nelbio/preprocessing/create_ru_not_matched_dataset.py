import codecs
import glob
import logging
import os.path
from argparse import ArgumentParser
import random

from nelbio.utils.io import load_dict


def main(args):
    """
    This script takes a directory to BioSyn-formatted dataset (with train/dev/test subsets) and dictionary as an input.
    It keeps mentions that have no CUI in the (Russian) vocabulary and writes them to a new BioSyn-formatted file.
    """
    random.seed(42)

    ru_vocab = load_dict(inp_path=args.vocab_path, sep="||")
    ru_vocab_cuis = set(ru_vocab.keys())

    output_dataset_dir = args.output_dataset_dir
    output_vocabs_dir = os.path.join(output_dataset_dir, "vocabs/")
    if not os.path.exists(output_vocabs_dir):
        os.makedirs(output_vocabs_dir)

    full_data_split_dir = os.path.join(output_dataset_dir, "datasets/full/")
    if not os.path.exists(full_data_split_dir):
        os.makedirs(full_data_split_dir)
    full_data_concept_path = os.path.join(full_data_split_dir, "0.concept")
    with codecs.open(full_data_concept_path, 'w+', encoding="utf-8") as out_full_file:
        for dataset_split_name in ("train", "dev", "test"):
            input_data_split_dir = os.path.join(args.input_dataset_dir, f"{dataset_split_name}/")
            output_data_split_dir = os.path.join(output_dataset_dir, f"datasets/{dataset_split_name}/")
            if not os.path.exists(output_data_split_dir):
                os.makedirs(output_data_split_dir)

            concept_paths = glob.glob(os.path.join(input_data_split_dir, "*.concept"))
            for concept_file_path in concept_paths:
                concept_name = concept_file_path.split('/')[-1]
                output_concept_file_dir = os.path.join(output_data_split_dir, concept_name)
                with open(concept_file_path, 'r', encoding="utf-8") as inp_file, \
                        codecs.open(output_concept_file_dir, 'w+', encoding="utf-8") as out_subset_file:
                    for line in inp_file:
                        attrs = line.strip().split('||')
                        cui = attrs[-1]

                        assert cui.startswith('C')
                        # If CUI is present in Russian part of UMLS, skip it
                        if cui in ru_vocab_cuis:
                            pass
                        else:
                            out_full_file.write(line)
                            out_subset_file.write(line)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--input_dataset_dir', type=str,
                        help="Path to BioSyn-formatted dataset split into train/dev/test")
    parser.add_argument('--vocab_path', type=str,
                        help="Path to BioSyn-formatted vocabulary")
    parser.add_argument('--output_dataset_dir', type=str,
                        help="Path to output directory")
    arguments = parser.parse_args()
    main(arguments)
