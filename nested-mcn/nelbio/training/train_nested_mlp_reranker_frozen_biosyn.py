import argparse
import logging
import os
import random
import time
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from nelbio.data.nested_datasets import NestedQueryDataset
from nelbio.data.nested_sep_flat_dataset import NestedSepFlatCandidateDataset
from nelbio.models.nested_biosyn import NestedBioSyn
from nelbio.models.nested_mlp_reranker import MLPReranker
from nelbio.utils.io import load_dictionary_tuples, save_dict, load_biosyn_formated_sep_context_dataset

LOGGER = logging.getLogger()


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Biosyn train')

    parser.add_argument('--pretrained_biosyn_path', required=True,
                        help='Directory for pretrained model')
    parser.add_argument('--train_dictionary_path', type=str, required=True,
                        help='train dictionary path')
    parser.add_argument('--train_dir', type=str, required=True,
                        help='training set directory')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output')
    #
    # # Tokenizer settings
    parser.add_argument('--query_max_length', default=25, type=int)
    parser.add_argument('--context_max_length', default=128, type=int)
    parser.add_argument('--initial_sparse_weight', default=1., type=float, required=False)
    parser.add_argument('--initial_flat_dense_weight', default=1., type=float, required=False)
    parser.add_argument('--initial_sep_dense_weight', default=1., type=float, required=False)


    parser.add_argument('--dense_encoder_score_type', type=str, choices=("matmul", "cosine"), default="cosine")
    parser.add_argument('--drop_not_nested', default=True)
    parser.add_argument("--keep_longest_entity_only", action="store_true")
    parser.add_argument("--drop_cuiless", action="store_true")
    parser.add_argument("--sep_pooling", choices=("cls", "mean"), default="cls")
    # Train config
    parser.add_argument('--seed', type=int,
                        default=0)
    parser.add_argument('--use_cuda', action="store_true")
    parser.add_argument('--draft', action="store_true")
    parser.add_argument('--topk', type=int,
                        default=20)
    parser.add_argument('--learning_rate',
                        help='learning rate',
                        default=0.0001, type=float)
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        default=0.01, type=float)
    parser.add_argument('--train_batch_size',
                        help='train batch size',
                        default=16, type=int)
    parser.add_argument('--epoch',
                        help='epoch to train',
                        default=10, type=int)
    parser.add_argument('--dense_ratio', type=float,
                        default=0.5)
    parser.add_argument('--save_checkpoint_all', action="store_true")

    args = parser.parse_args()
    return args


def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)


def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_queries(data_dir, filter_composite, filter_cuiless, pad_nested, drop_not_nested, cuis_vocab):
    """
    load query data

    Parameters
    ----------
    is_train : bool
        train or dev
    filter_composite : bool
        filter composite mentions
    filter_cuiless : bool
        filter samples with cuiless
    """
    dataset = NestedQueryDataset(
        data_dir=data_dir,
        filter_composite=filter_composite,
        filter_cuiless=filter_cuiless,
        pad_nested=pad_nested,
        drop_not_nested=drop_not_nested,
        cuis_vocab=cuis_vocab
    )
    n_m = dataset.nested_entity_mentions
    n_cuis = dataset.nested_entity_cuis
    n_c_masks = dataset.nested_contribution_masks
    n_p_masks = dataset.entity_padding_masks
    max_n_depth = dataset.max_nesting_depth

    return n_m, n_cuis, n_c_masks, n_p_masks, max_n_depth


def create_nested_entity_flat_index(nested_entities: List[List[str]]):
    index = []
    start_pos = 0
    for nested_e_id, n_e in enumerate(nested_entities):
        depth = len(n_e)

        end_pos = start_pos + depth
        index.append((start_pos, end_pos))
        start_pos += depth
    return index


def train(args, data_loader, model, device):
    LOGGER.info("train!")
    train_loss = 0
    # bert_tokenizer = data_loader.dataset.tokenizer
    train_steps = 0
    model.train()
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.optimizer.zero_grad()

        query_token_inp_ids = batch["query_token"]["input_ids"].to(device)
        query_token_att_mask = batch["query_token"]["attention_mask"].to(device)
        sep_context_inp_ids = batch["context_input"]["input_ids"].to(device)

        sep_context_att_mask = batch["context_input"]["attention_mask"].to(device)
        candidate_token_inp_ids = batch["candidate_token"]["input_ids"].to(device)
        candidate_att_mask = batch["candidate_token"]["attention_mask"].to(device)
        nested_sep_candidate_input_ids = batch["nested_sep_candidate_input"]["input_ids"].to(device)
        nested_sep_candidate_att_mask = batch["nested_sep_candidate_input"]["attention_mask"].to(device)
        nested_mask = batch["nested_mask"].to(device)

        query_token_input = (query_token_inp_ids, query_token_att_mask)
        candidate_input = (candidate_token_inp_ids, candidate_att_mask)
        sep_context_input = (sep_context_inp_ids, sep_context_att_mask)
        nested_sep_candidate_input = (nested_sep_candidate_input_ids, nested_sep_candidate_att_mask)
        candidate_s_scores = batch["candidate_s_scores"].to(device)
        batch_y = batch["labels"]

        batch_pred = model(query_token_input=query_token_input, candidate_input=candidate_input,
                           sep_context_input=sep_context_input, nested_sep_candidate_input=nested_sep_candidate_input,
                           candidate_s_scores=candidate_s_scores, nested_mask=nested_mask)
        loss = model.get_loss(batch_pred, batch_y, device=device)
        loss.backward()
        model.optimizer.step()
        train_loss += loss.item()
        train_steps += 1

    train_loss /= (train_steps + 1e-9)
    return train_loss


def main(args):
    init_logging()
    init_seed(args.seed)
    print(args)

    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"

    # prepare for output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    experiment_config_path = os.path.join(args.output_dir, "experiment_config.txt")
    save_dict(d=vars(args), save_path=experiment_config_path)

    # load dictionary and queries
    train_dictionary = load_dictionary_tuples(inp_path=args.train_dictionary_path)
    train_vocab_cuis_set = set(train_dictionary[:, 1])


    # filter only names
    names_in_train_dictionary = train_dictionary[:, 0]

    # load BERT tokenizer, dense_encoder, sparse_encoder
    biosyn = NestedBioSyn(
        max_length=args.query_max_length,
        use_cuda=args.use_cuda,
        initial_sparse_weight=args.initial_sparse_weight,
    )
    biosyn.load_model(model_name_or_path=args.pretrained_biosyn_path,)
    for param in biosyn.get_dense_encoder().parameters():
        param.requires_grad = False
    biosyn.sparse_weight.requires_grad = False

    model = MLPReranker(bert_encoder=biosyn.get_dense_encoder(),
                                tokenizer=biosyn.get_dense_tokenizer(),
                                sparse_encoder=biosyn.get_sparse_encoder(),
                                learning_rate=args.learning_rate,
                                mlp_lr=1e-3,
                                weight_decay=args.weight_decay,
                                sparse_weight=biosyn.sparse_weight,
                                dense_encoder_score_type=args.dense_encoder_score_type,
                                sep_pooling="cls",
                                ).to(device)
    sep_contexts, flat_cuis, flat_queries, n_m_list = \
        load_biosyn_formated_sep_context_dataset(args.train_dir,
                                                 drop_cuiless=args.drop_cuiless,
                                                 cui_dictionary=train_vocab_cuis_set,
                                                 drop_duplicates=True,
                                                 drop_not_nested=True)
    sep_token = biosyn.get_dense_tokenizer().sep_token
    if args.keep_longest_entity_only:
        new_sep_contexts = []
        new_n_m_list = []
        for sp, nm, fq in zip(sep_contexts, n_m_list, flat_queries):
            if len(nm) == 1:
                new_sep_contexts.append(sp)
                new_n_m_list.append(nm)
            else:
                longest_entity = max(nm, key=lambda t: len(t))
                new_sp = f"{fq} {sep_token} {longest_entity}"
                new_nm = [fq, longest_entity]
                new_sep_contexts.append(new_sp)
                new_n_m_list.append(new_nm)
        sep_contexts = new_sep_contexts
        n_m_list = new_n_m_list

    train_set = NestedSepFlatCandidateDataset(
        sep_contexts=sep_contexts,
        flat_cuis=flat_cuis,
        flat_queries=flat_queries,
        n_m_list=n_m_list,
        dictionary=train_dictionary,
        tokenizer=biosyn.get_dense_tokenizer(),
        query_max_length=args.query_max_length,
        context_max_length=args.context_max_length,
        topk=args.topk,
        d_ratio=args.dense_ratio,
        s_score_matrix=None,
        s_candidate_idxs=None)

    flat_mentions = train_set.flat_queries
    num_flat_e = len(train_set)

    # <num_entities, entity_depth, vocab_size>
    sparse_score_matrix = \
        biosyn.get_torch_query_dict_score_matrix(query_names=flat_mentions,
                                                 vocab_names=names_in_train_dictionary,
                                                 device=device,
                                                 vocab_batch_size=32,
                                                 dense_encoder_score_type=args.dense_encoder_score_type,
                                                 show_progress=True)
    LOGGER.info(f"sparse_score_matrix.shape {sparse_score_matrix.shape}")
    # train_sparse_score_matrix.reshape(shape=(num_entities, entity_depth, sparse_emb_size))
    train_s_candidate_idxs = biosyn.retrieve_candidate(
        score_matrix=sparse_score_matrix,
        topk=args.topk
    )
    train_set.set_s_score_matrix(sparse_score_matrix)
    train_set.set_s_candidate_idxs(train_s_candidate_idxs)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=train_set.collate_fn
    )

    start = time.time()
    for epoch in range(1, args.epoch + 1):
        # embed dense representations for query and dictionary for train
        # Important! This is iterative process because dense represenation changes as model is trained.
        LOGGER.info("Epoch {}/{}".format(epoch, args.epoch))
        LOGGER.info("train_set dense embedding for iterative candidate retrieval")
        train_query_dense_embeds = biosyn.embed_dense(names=flat_mentions, show_progress=True)
        train_dict_dense_embeds = biosyn.embed_dense(names=names_in_train_dictionary, show_progress=True)

        train_dense_score_matrix = biosyn.get_score_matrix(
            query_embeds=train_query_dense_embeds,
            dict_embeds=train_dict_dense_embeds,
            dense_encoder_score_type=args.dense_encoder_score_type
        )
        # <batch_size, topk>
        train_d_candidate_idxs = biosyn.retrieve_candidate(
            score_matrix=train_dense_score_matrix,
            topk=args.topk
        )
        train_set.set_dense_candidate_idxs(d_candidate_idxs=train_d_candidate_idxs)

        # train
        train_loss = train(args, data_loader=train_loader, model=model, device=device)
        LOGGER.info(f"loss/train_per_epoch={train_loss}/{epoch}")

        # save model every epoch
        if args.save_checkpoint_all:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint_{}".format(epoch))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model.save_model(checkpoint_dir)

        # save model last epoch
        if epoch == args.epoch:
            model.save_model(args.output_dir)

    end = time.time()
    training_time = end - start
    training_hour = int(training_time / 60 / 60)
    training_minute = int(training_time / 60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info("Training Time!{} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))


if __name__ == '__main__':
    args = parse_args()
    main(args)
