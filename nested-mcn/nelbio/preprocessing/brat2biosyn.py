import logging
from argparse import ArgumentParser

from nelbio.preprocessing.utils import ANNOTATION_FILENAME_PATTERNS, convert_brat_dirs_to_biosyn


def main(args):
    input_directories = args.input_directories
    ann_filename_pattern = ANNOTATION_FILENAME_PATTERNS[args.annotation_filename_pattern]
    output_dir = args.output_dir

    drop_cuiless = args.drop_cuiless

    convert_brat_dirs_to_biosyn(dir_list=input_directories, ann_filename_pattern=ann_filename_pattern,
                                drop_cuiless=drop_cuiless, output_dir=output_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--input_directories', nargs='+', type=str,
                        help="A list to directories with BRAT-formatted annotations ")
    parser.add_argument('--drop_cuiless', action="store_true")
    parser.add_argument('--annotation_filename_pattern', choices=("en", "ru"), type=str, default="ru")
    parser.add_argument('--output_dir', type=str)

    arguments = parser.parse_args()
    main(arguments)
