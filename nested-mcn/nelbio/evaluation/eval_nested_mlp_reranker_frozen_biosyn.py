import argparse
import json
import logging
import os

import torch
from transformers import AutoModel, AutoTokenizer

from nelbio.data.nested_sep_flat_dataset import NestedSepEvaluationQueryDataset
from nelbio.models.nested_biosyn import NestedBioSyn
from nelbio.models.nested_mlp_reranker import MLPReranker
from nelbio.utils.io import load_dict, load_dictionary_tuples, load_biosyn_formated_sep_context_dataset
from nelbio.utils.nested_utils_sep_context_mlp_reranker import evaluate_nested_flat_reranker

LOGGER = logging.getLogger()


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='BioSyn evaluation')

    parser.add_argument('--model_name_or_path', required=True, help='Directory for model')
    parser.add_argument('--model_config_path', required=True, help='Directory for model')
    parser.add_argument('--dictionary_path', type=str, required=True, help='dictionary path')
    parser.add_argument('--data_dir', type=str, required=True, help='data set to evaluate')

    # Run settings
    parser.add_argument('--use_cuda', action="store_true")
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--score_mode', type=str, default='hybrid', help='hybrid/dense/sparse')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Directory for output')
    parser.add_argument('--filter_composite', action="store_true", help="filter out composite mention queries")
    parser.add_argument('--filter_duplicate', action="store_true", help="filter out duplicate queries")
    parser.add_argument('--save_predictions', action="store_true", help="whether to save predictions")
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--drop_not_nested', action="store_true", help="Drops all non-nested entities")
    parser.add_argument('--dense_encoder_score_type', type=str, choices=("matmul", "cosine"), required=False)
    parser.add_argument('--drop_cuiless', action="store_true")

    parser.add_argument('--force_flat', action="store_true")

    parser.add_argument('--query_max_length', default=25, type=int)
    parser.add_argument('--context_max_length', default=128, type=int)

    args = parser.parse_args()
    return args


def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)


def main(args):
    init_logging()
    print(args)
    if not os.path.exists(args.output_dir) and args.output_dir != '':
        os.makedirs(args.output_dir)
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"

    experiment_cfg_dict = load_dict(args.model_config_path)

    keep_longest_entity_only = experiment_cfg_dict.get("keep_longest_entity_only", "False")
    keep_longest_entity_only = True if keep_longest_entity_only.strip() == "True" else False
    sep_pooling = experiment_cfg_dict.get("sep_pooling", "cls")
    force_flat_flag = args.force_flat

    # load dictionary and data
    eval_dictionary = load_dictionary_tuples(inp_path=args.dictionary_path)
    eval_vocab_cuis_set = set(eval_dictionary[:, 1])
    print(experiment_cfg_dict)

    dense_encoder_score_type = experiment_cfg_dict.get("dense_encoder_score_type")
    if dense_encoder_score_type is None:
        dense_encoder_score_type = args.dense_encoder_score_type
        logging.info(f"Could not find dense_encoder_score_type in experiment's config,"
                     f"using value from argparser: {args.dense_encoder_score_type}")
    assert dense_encoder_score_type in ("matmul", "cosine")

    bert_encoder = AutoModel.from_pretrained(args.model_name_or_path)
    bert_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    context_sep_token = bert_tokenizer.sep_token

    biosyn = NestedBioSyn(
        max_length=args.query_max_length,
        use_cuda=args.use_cuda,
    )
    biosyn.load_model(
        model_name_or_path=args.model_name_or_path,
    )
    sep_contexts, flat_cuis, flat_queries, n_m_list = \
        load_biosyn_formated_sep_context_dataset(args.data_dir,
                                                 cui_dictionary=eval_vocab_cuis_set,
                                                 drop_duplicates=True,
                                                 drop_not_nested=True,
                                                 drop_cuiless=args.drop_cuiless)
    if keep_longest_entity_only:
        new_sep_contexts = []
        new_n_m_list = []
        for sp, nm, fq in zip(sep_contexts, n_m_list, flat_queries):
            if len(nm) == 1:
                new_sep_contexts.append(sp)
                new_n_m_list.append(nm)
            else:
                longest_entity = max(nm, key=lambda t: len(t))
                new_sp = f"{fq} {context_sep_token} {longest_entity}"
                new_nm = [fq, longest_entity]
                new_sep_contexts.append(new_sp)
                new_n_m_list.append(new_nm)
        sep_contexts = new_sep_contexts
        n_m_list = new_n_m_list

    eval_query_dataset = NestedSepEvaluationQueryDataset(
        sep_contexts=sep_contexts,
        flat_queries=flat_queries,
        flat_cuis=flat_cuis,
        n_m_list=n_m_list,
        tokenizer=biosyn.tokenizer,
        query_max_length=args.query_max_length,
        context_max_length=args.context_max_length,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_query_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=eval_query_dataset.collate_fn
    )

    model = MLPReranker(bert_encoder,
                        tokenizer=bert_tokenizer,
                        sparse_encoder=biosyn.sparse_encoder,
                        learning_rate=1e-5,
                        weight_decay=0.01,
                        sparse_weight=biosyn.get_sparse_weight(),
                        dense_encoder_score_type=args.dense_encoder_score_type,
                        sep_pooling=sep_pooling,
                        mlp_lr=1e-3).to(device)
    model.load_model(args.model_name_or_path)

    result_evalset = evaluate_nested_flat_reranker(
        eval_dataloader=eval_dataloader,
        model=model,
        bert_tokenizer=bert_tokenizer,
        filter_cuiless=True,
        dense_encoder_score_type=dense_encoder_score_type,
        biosyn=biosyn,
        eval_dictionary=eval_dictionary,
        topk=args.topk,
        score_mode=args.score_mode,
        device=device,
        context_max_length=args.context_max_length,
        context_sep_token=context_sep_token,
        log_dir=args.output_dir,
        sep_pooling=sep_pooling,
        force_flat_flag=force_flat_flag
    )

    LOGGER.info("acc@1={}".format(result_evalset['acc1']))
    LOGGER.info("acc@5={}".format(result_evalset['acc5']))

    if args.save_predictions:
        output_file = os.path.join(args.output_dir, "predictions_eval.json")
        with open(output_file, 'w') as f:
            json.dump(result_evalset, f, indent=2)


if __name__ == '__main__':
    args = parse_args()
    main(args)
