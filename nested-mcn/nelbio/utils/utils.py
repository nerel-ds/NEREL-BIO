import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import default_data_collator

from nelbio.models.nested_biosyn import NamesDataset


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


def predict_topk(biosyn, eval_dictionary, eval_queries, topk, log_dir: str, score_mode='hybrid'):
    """
    Parameters
    ----------
    score_mode : str
        hybrid, dense, sparse
    """
    encoder = biosyn.get_dense_encoder()
    tokenizer = biosyn.get_dense_tokenizer()
    sparse_encoder = biosyn.get_sparse_encoder()
    sparse_weight = biosyn.get_sparse_weight().item()  # must be scalar value

    # embed dictionary
    dict_sparse_embeds = biosyn.embed_sparse(names=eval_dictionary[:, 0], show_progress=True)
    dict_dense_embeds = biosyn.embed_dense(names=eval_dictionary[:, 0], show_progress=True)
    top_1_predictions_file_path = os.path.join(log_dir, "top1_predictions.txt")

    deleted_cuis_count = 0
    cuiless_count = 0
    queries = []
    with open(top_1_predictions_file_path, "w+", encoding="utf-8") as out_top1_file:
        for eval_query in tqdm(eval_queries, total=len(eval_queries)):
            mentions = eval_query[0].replace("+", "|").split("|")
            golden_cui = eval_query[1].replace("+", "|")
            if golden_cui == "-D":
                deleted_cuis_count += 1
                continue
            if golden_cui == "-1":
                cuiless_count += 1
                continue

            dict_mentions = []
            for mention in mentions:
                mention_sparse_embeds = biosyn.embed_sparse(names=np.array([mention]))
                mention_dense_embeds = biosyn.embed_dense(names=np.array([mention]))

                # get score matrix
                sparse_score_matrix = biosyn.get_score_matrix(
                    query_embeds=mention_sparse_embeds,
                    dict_embeds=dict_sparse_embeds
                )
                dense_score_matrix = biosyn.get_score_matrix(
                    query_embeds=mention_dense_embeds,
                    dict_embeds=dict_dense_embeds
                )
                if score_mode == 'hybrid':
                    score_matrix = sparse_weight * sparse_score_matrix + dense_score_matrix
                elif score_mode == 'dense':
                    score_matrix = dense_score_matrix
                elif score_mode == 'sparse':
                    score_matrix = sparse_score_matrix
                else:
                    raise NotImplementedError()

                candidate_idxs = biosyn.retrieve_candidate(
                    score_matrix=score_matrix,
                    topk=topk
                )
                np_candidates = eval_dictionary[candidate_idxs].squeeze()
                top_1_cand = eval_dictionary[candidate_idxs[0]]
                top_1_label = check_label(np_candidates[0][1], golden_cui)  # top_1_cand[1]
                out_top1_file.write(
                    f"{mention} ({golden_cui})\t{top_1_cand[0]} ({top_1_cand[1]})\t{top_1_label}\n")
                dict_candidates = []
                for np_candidate in np_candidates:
                    dict_candidates.append({
                        'name': np_candidate[0],
                        'cui': np_candidate[1],
                        'label': check_label(np_candidate[1], golden_cui)
                    })
                dict_mentions.append({
                    'mention': mention,
                    'golden_cui': golden_cui,  # golden_cui can be composite cui
                    'candidates': dict_candidates
                })
            queries.append({
                'mentions': dict_mentions
            })
    result = {
        'queries': queries
    }
    logging.info(f"Evaluation is finished. num queries: {len(queries)}\n"
                 f"There are {cuiless_count} CUIless mentions."
                 f"{deleted_cuis_count} CUIs have been deleted (marked with '-D')")

    return result


def evaluate(biosyn, eval_dictionary, eval_queries, topk, log_dir, score_mode='hybrid'):
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
    result = predict_topk(biosyn, eval_dictionary, eval_queries, topk, log_dir, score_mode)
    result = evaluate_topk_acc(result)

    return result


def bert_embed_dense(bert_encoder, bert_tokenizer, max_length, names, device, return_type, show_progress=False):
    """
    Embedding data into dense representations

    Parameters
    ----------
    names : np.array or list
        An array of names

    Returns
    -------
    dense_embeds : list
        A list of dense embeddings
    """
    assert return_type in ("torch", "numpy")
    bert_encoder.eval()  # prevent dropout

    batch_size = 1024
    dense_embeds = []

    if isinstance(names, np.ndarray):
        names = names.tolist()
    name_encodings = bert_tokenizer(names, padding="max_length", max_length=max_length,
                                    return_token_type_ids=True, truncation=True, return_tensors="pt")
    name_encodings = name_encodings.to(device)
    name_dataset = NamesDataset(name_encodings)
    name_dataloader = DataLoader(name_dataset, shuffle=False, collate_fn=default_data_collator,
                                 batch_size=batch_size)

    with torch.no_grad():
        for batch in tqdm(name_dataloader, disable=not show_progress, desc='embedding dictionary'):
            outputs = bert_encoder(**batch)
            batch_dense_embeds = outputs[0][:, 0].cpu().detach()  # [CLS] representations
            if return_type == "numpy":
                batch_dense_embeds = batch_dense_embeds.numpy()
            dense_embeds.append(batch_dense_embeds)
    if return_type == "numpy":
        dense_embeds = np.concatenate(dense_embeds, axis=0)
    elif return_type == "torch":
        dense_embeds = torch.cat(dense_embeds, dim=0)

    return dense_embeds


def retrieve_candidate(score_matrix, topk):
    """
    Return sorted topk idxes (descending order)
    Parameters
    ----------
    score_matrix : np.array
        2d numpy array of scores
    topk : int
        The number of candidates
    Returns
    -------
    topk_idxs : np.array
        2d numpy array of scores [# of query , # of dict]
    """

    def indexing_2d(arr, cols):
        rows = np.repeat(np.arange(0, cols.shape[0])[:, np.newaxis], cols.shape[1], axis=1)
        return arr[rows, cols]

    # get topk indexes without sorting
    topk_idxs = np.argpartition(score_matrix, -topk)[:, -topk:]

    # get topk indexes with sorting
    topk_score_matrix = indexing_2d(score_matrix, topk_idxs)
    topk_argidxs = np.argsort(-topk_score_matrix)
    topk_idxs = indexing_2d(topk_idxs, topk_argidxs)

    return topk_idxs


def get_score_matrix(query_embeds: np.array, dict_embeds: np.array, score_type="matmul"):
    """
    Return score matrix
    Parameters
    ----------
    query_embeds : np.array <num_queries, emb_size>
        2d numpy array of query embeddings
    dict_embeds : np.array <dict_size, emb_size>
        2d numpy array of query embeddings
    Returns
    -------
    score_matrix : np.array
        2d numpy array of scores
    """

    if score_type == "matmul":
        score_matrix = np.matmul(query_embeds, dict_embeds.T)
    elif score_type == "cosine":
        score_matrix = cosine_similarity(query_embeds, dict_embeds)
    else:
        raise Exception(f"Invalid dense_encoder_score_type: {score_type}")

    return score_matrix


def marginal_nll_no_mask(score, target):
    """
    sum all scores among positive samples
    """
    predict = F.softmax(score, dim=-1)
    loss = predict * target
    loss = loss.sum(dim=-1)  # sum all positive scores
    loss = loss[loss > 0]  # filter sets with at least one positives
    loss = torch.clamp(loss, min=1e-9, max=1)  # for numerical stability
    loss = -torch.log(loss)  # for negative log likelihood
    if len(loss) == 0:
        loss = loss.sum()  # will return zero loss
    else:
        loss = loss.mean()
    return loss


def add_bert_tokens(tokens, bert_model, bert_tokenizer, ):
    num_added_tokens = bert_tokenizer.add_tokens(tokens)
    bert_model.resize_token_embeddings(len(bert_tokenizer))
    print('Number of tokens added:', num_added_tokens)
