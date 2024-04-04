import logging
import os.path
from typing import Union, List

import numpy
import numpy as np
import torch
from torch.utils.data import DataLoader
from nelbio.utils.utils import get_score_matrix


def check_k(queries):
    return len(queries[0]['mentions'][0]['candidates'])


def evaluate_topk_acc(data):
    """
    evaluate acc@1~acc@k
    """
    queries = data['queries']
    k = check_k(queries)

    for i in range(0, k):
        hit = 0
        for query in queries:
            mentions = query['mentions']
            mention_hit = 0
            for mention in mentions:
                candidates = mention['candidates'][:i + 1]  # to get acc@(i+1)
                mention_hit += np.any([candidate['label'] for candidate in candidates])

            # When all mentions in a query are predicted correctly,
            # we consider it as a hit 
            if mention_hit == len(mentions):
                hit += 1

        data['acc{}'.format(i + 1)] = hit / len(queries)

    return data


def check_label(predicted_cui, golden_cui):
    """
    Some composite annotation didn't consider orders
    So, set label '1' if any cui is matched within composite cui (or single cui)
    Otherwise, set label '0'
    """
    return int(len(set(predicted_cui.split("|")).intersection(set(golden_cui.split("|")))) > 0)


def get_flat_score_matrix(score_mode, sparse_weight, dense_weight, flat_sparse_score_matrix, flat_dense_score_matrix):
    if score_mode == 'hybrid':
        flat_score_matrix = sparse_weight * flat_sparse_score_matrix + flat_dense_score_matrix * dense_weight
    elif score_mode == 'dense':
        flat_score_matrix = flat_dense_score_matrix * dense_weight
    elif score_mode == 'sparse':
        flat_score_matrix = flat_sparse_score_matrix
    else:
        raise NotImplementedError()
    return flat_score_matrix


def label_predicted_candidates(candidates_idx, vocab, mention, golden_cui):
    np_candidates = vocab[candidates_idx].squeeze()
    dict_candidates = []
    for np_candidate in np_candidates:
        dict_candidates.append({
            'name': np_candidate[0],
            'cui': np_candidate[1],
            'label': check_label(np_candidate[1], golden_cui)
        })
    dict_mentions = [{
        'mention': mention,
        'golden_cui': golden_cui,  # golden_cui can be composite cui
        'candidates': dict_candidates}]

    return dict_mentions


def predict_topk(eval_dataloader: DataLoader, filter_cuiless, dense_encoder_score_type, model,
                 biosyn, bert_tokenizer, eval_dictionary, topk, score_mode, context_max_length, context_sep_token,
                 sep_pooling, force_flat_flag, log_dir, device):
    """
    Parameters
    ----------
    score_mode : str
        hybrid, dense, sparse
    """
    bert_encoder = biosyn.get_dense_encoder()

    sparse_weight = model.sparse_weight.item()  # must be scalar value
    flat_dense_weight = model.flat_dense_weight.item()
    sep_dense_weight = model.sep_dense_weight.item()
    if force_flat_flag:
        sep_dense_weight = 0.
    if score_mode == "dense":
        sparse_weight = 0.
    # embed dictionary
    dict_sparse_embeds = biosyn.embed_sparse(names=eval_dictionary[:, 0], show_progress=True)
    # <vocab_size, emb_size>
    dict_dense_embeds = biosyn.embed_dense(names=eval_dictionary[:, 0], show_progress=True)
    bert_encoder.eval()
    model.eval()
    cuiless_count = 0
    deleted_cuis_count = 0
    queries = []
    log_file_path = os.path.join(log_dir, "reranking_examples.txt")
    mismatch_log_file_path = os.path.join(log_dir, "mismatched_reranking_examples.txt")
    top_1_predictions_file_path = os.path.join(log_dir, "top1_predictions.txt")
    logging.info(f"Using sep pooling: {sep_pooling}")
    with open(log_file_path, "w+", encoding="utf-8") as out_file, \
            open(mismatch_log_file_path, "w+", encoding="utf-8") as out_mismatch_file, \
            open(top_1_predictions_file_path, "w+", encoding="utf-8") as out_top1_file:

        for batch in eval_dataloader:
            query_token, mentions, nested_mentions, sep_context_token, cuis = (batch)
            flat_mention_sparse_embeds = biosyn.embed_sparse(np.array(mentions)).detach().cpu().numpy()
            query_token_inp_ids = query_token["input_ids"].squeeze(1).to(device)
            query_token_att_mask = query_token["attention_mask"].squeeze(1).to(device)

            sep_context_inp_ids = sep_context_token["input_ids"].squeeze(1).to(device)
            sep_context_att_mask = sep_context_token["attention_mask"].squeeze(1).to(device)

            with (torch.no_grad()):
                flat_query_emb = model.bert_encode(query_token_inp_ids, query_token_att_mask)
                sep_context_emb = model.bert_encode(sep_context_inp_ids, sep_context_att_mask, pooling=sep_pooling)

                # <flat_batch_size, vocab_size>
                sparse_score_matrix = get_score_matrix(
                    query_embeds=flat_mention_sparse_embeds,
                    dict_embeds=dict_sparse_embeds,
                    score_type=dense_encoder_score_type

                )
                # <flat_batch_size, vocab_size>
                flat_dense_score_matrix = get_score_matrix(
                    query_embeds=flat_query_emb.detach().cpu().numpy(),
                    dict_embeds=dict_dense_embeds,
                    score_type=dense_encoder_score_type
                )

                flat_score_matrix = get_flat_score_matrix(score_mode, sparse_weight, flat_dense_weight,
                                                          sparse_score_matrix,
                                                          flat_dense_score_matrix)
                # <b, k>
                candidate_idxs = biosyn.retrieve_candidate(
                    score_matrix=flat_score_matrix,
                    topk=topk,
                )
                batch_size, k = candidate_idxs.shape
                assert k == topk
                # <b, k>
                cand_nested_mentions = eval_dictionary[candidate_idxs.reshape((-1,)), 0].reshape((batch_size, k,))
                # print("cand_nested_mentions", cand_nested_mentions)
                cand_nested_sep_contexts = []
                for i, c_m_list in enumerate(cand_nested_mentions):
                    q_mention: List[str] = nested_mentions[i]
                    for c_m in c_m_list:
                        # sep_candidate = [sc for sc in q_mention if sc != c_m]
                        sep_candidate = q_mention.copy()
                        sep_candidate[0] = c_m
                        sep_candidate = context_sep_token.join(sep_candidate)
                        cand_nested_sep_contexts.append(sep_candidate)
                        # print("mention", mentions[i], "||| sep_candidate", sep_candidate)
                    # print('-')
                assert len(cand_nested_sep_contexts) == batch_size * k
                sep_context_token = bert_tokenizer(cand_nested_sep_contexts, max_length=context_max_length,
                                                   return_token_type_ids=False, padding='max_length', truncation=True,
                                                   return_tensors='pt')
                sep_context_token_inp_ids = sep_context_token["input_ids"].to(device)
                sep_context_token_att_mask = sep_context_token["attention_mask"].to(device)
                # <b * k, e>
                sep_candidate_emb = model.bert_encode(sep_context_token_inp_ids, sep_context_token_att_mask,
                                                      pooling=sep_pooling
                                                      ).view((batch_size, k, -1))

                sep_dense_score_matrix = model.calculate_dense_scores(query_emb=sep_context_emb,
                                                                      candidate_emb=sep_candidate_emb
                                                                      ).detach().cpu().numpy()
                sub_sparse_score_matrix = numpy.zeros(shape=(batch_size, topk), dtype=np.float32)
                sub_flat_dense_score_matrix = numpy.zeros(shape=(batch_size, topk), dtype=np.float32)

                for i in range(batch_size):
                    sub_sparse_score_matrix[i, :] = sparse_score_matrix[i][candidate_idxs[i]]
                    sub_flat_dense_score_matrix[i, :] = flat_dense_score_matrix[i][candidate_idxs[i]]
                sum_score_matrix = sparse_weight * sub_sparse_score_matrix + flat_dense_weight * \
                                   sub_flat_dense_score_matrix + sep_dense_weight * sep_dense_score_matrix

                idxs_over_candidate_idxs = biosyn.retrieve_candidate(
                    score_matrix=sum_score_matrix,
                    topk=topk,
                )

                for i, (cand_idx_row, idx_over_idx) in enumerate(zip(candidate_idxs, idxs_over_candidate_idxs)):
                    if len(nested_mentions[i]) == 1:
                        continue
                    out_file.write(
                        f"Mention: <{mentions[i]}> ({cuis[i]}), context: <{context_sep_token.join(nested_mentions[i])}>:\n")
                    candidate_mentions_before = tuple(eval_dictionary[candidate_idxs[i], 0])
                    candidate_cuis_before = tuple(eval_dictionary[candidate_idxs[i], 1])
                    cnd_tuple_strs_before = "||".join(
                        (f"{x} ({y})" for x, y in zip(candidate_mentions_before, candidate_cuis_before)))
                    out_file.write(f"\tRanking before: {cnd_tuple_strs_before}\n")

                    top_1_cand_before = eval_dictionary[candidate_idxs[i][0], 0]
                    # print(f"{i + 1} { mentions[i]} BEFORE", candidate_idxs[i], eval_dictionary[candidate_idxs[i], 0] )
                    candidate_idxs[i, :] = candidate_idxs[i][idx_over_idx]
                    # print(f"{i + 1} { mentions[i]} AFTER ", candidate_idxs[i], eval_dictionary[candidate_idxs[i], 0])
                    candidate_mentions_after = tuple(eval_dictionary[candidate_idxs[i], 0])
                    candidate_cuis_after = tuple(eval_dictionary[candidate_idxs[i], 1])
                    cnd_tuple_strs_after = "||".join(
                        (f"{x} ({y})" for x, y in zip(candidate_mentions_after, candidate_cuis_after)))
                    out_file.write(f"\tRanking after: {cnd_tuple_strs_after}\n--\n")
                    top_1_cand_after = eval_dictionary[candidate_idxs[i][0], 0]
                    if top_1_cand_before != top_1_cand_after:
                        out_mismatch_file \
                            .write(
                            f"Mention: <{mentions[i]}> ({cuis[i]}), context: <{context_sep_token.join(nested_mentions[i])}>:\n")
                        out_mismatch_file.write(
                            f"\tRanking before: {cnd_tuple_strs_before}\n\tRanking after: {cnd_tuple_strs_after}\n\--\n")

                for i, golden_cui in enumerate(cuis):
                    top_1_cand = eval_dictionary[candidate_idxs[i][0]]
                    top_1_label = check_label(top_1_cand[1], golden_cui)
                    out_top1_file.write(
                        f"{mentions[i]} ({golden_cui})\t{top_1_cand[0]} ({top_1_cand[1]})\t{top_1_label}\n")
                    if filter_cuiless:
                        if golden_cui == "-1":
                            cuiless_count += 1
                            continue
                        if golden_cui == "-D":
                            deleted_cuis_count += 1
                            continue
                    dict_mentions = label_predicted_candidates(candidates_idx=candidate_idxs[i], vocab=eval_dictionary,
                                                               mention=mentions[i], golden_cui=golden_cui)

                    queries.append({
                        'mentions': dict_mentions
                    })
                # print('--')

    result = {
        'queries': queries
    }
    logging.info(f"Finished topk prediction. {cuiless_count} CUIless entities are dropped.\n"
                 f"Found {deleted_cuis_count} deleted cuis.\n"
                 f"Evaluation result length: {len(result['queries'])}")

    return result


def evaluate_nested_flat_with_sep_context(eval_dataloader: DataLoader, model, biosyn, eval_dictionary, topk, score_mode,
                                          filter_cuiless, bert_tokenizer, context_max_length, dense_encoder_score_type,
                                          context_sep_token, log_dir, device, sep_pooling, force_flat_flag):
    """
    predict topk and evaluate accuracy
    
    Parameters
    ----------
    biosyn : BioSyn
        trained biosyn model
    eval_dictionary : str
        dictionary to evaluate
    eval_queries : str
        queries to evaluate
    topk : int
        the number of topk predictions
    score_mode : str
        hybrid, dense, sparse
    Returns
    -------
    result : dict
        accuracy and candidates
    """
    result = predict_topk(eval_dataloader=eval_dataloader,
                          filter_cuiless=filter_cuiless,
                          model=model,
                          bert_tokenizer=bert_tokenizer,
                          context_max_length=context_max_length,
                          dense_encoder_score_type=dense_encoder_score_type,
                          biosyn=biosyn,
                          eval_dictionary=eval_dictionary,
                          topk=topk,
                          score_mode=score_mode,
                          context_sep_token=context_sep_token,
                          sep_pooling=sep_pooling,
                          force_flat_flag=force_flat_flag,
                          log_dir=log_dir,
                          device=device, )
    result = evaluate_topk_acc(result)

    return result


def check_query_label(query_id, candidate_id_set):
    label = 0
    query_ids = query_id.split("|")
    """
    All query ids should be included in dictionary id
    """

    if (len(query_id) == 0) or (len(query_id) == 1 and query_id[0] == ''):
        return 0
    for q_id in query_ids:
        assert q_id != ''
        if q_id in candidate_id_set:
            label = 1
            continue
        else:
            label = 0
            break
    return label
