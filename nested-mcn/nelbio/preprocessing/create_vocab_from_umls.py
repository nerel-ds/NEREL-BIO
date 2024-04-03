import logging
import os
from argparse import ArgumentParser

from nelbio.utils.io import read_mrconso


def main(args):
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    logging.info(f"Loading MRCONSO....")
    mrconso_df = read_mrconso(fpath=args.mrconso)
    logging.info(f"Filtering MRCONSO....")
    mrconso_df = mrconso_df[mrconso_df.LAT.isin(args.langs)]
    cui_str_df = mrconso_df[["CUI", "STR"]]
    logging.info(f"Saving vocab....")
    cui_str_df.to_csv(output_path, sep='||', index=False, header=None)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--mrconso', help="Path to MRCONSO.RRF file of the UMLS metathesaurus.")
    parser.add_argument('--langs', nargs='+', default=['RUS', ])
    parser.add_argument('--output_path', default="../../data/dictionary/vocab_umls_rus_biosyn.txt")
    arguments = parser.parse_args()
    main(arguments)
