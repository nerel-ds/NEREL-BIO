import csv
import json
import numpy as np
import pdb

import torch
from tqdm import tqdm

from nelbio.models.nested_reranknet import NestedTransformerEncoder


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



def predict_topk(nested_encoder: NestedTransformerEncoder, biosyn, eval_dictionary, nested_eval_queries,
                 nested_entity_cuis, entity_padding_masks, topk, score_mode, device):
    """
    Parameters
    ----------
    score_mode : str
        hybrid, dense, sparse
    """
    bert_encoder = biosyn.get_dense_encoder()
    bert_tokenizer = biosyn.get_dense_tokenizer()
    max_length = biosyn.max_length
    sparse_encoder = biosyn.get_sparse_encoder()
    sparse_weight = biosyn.get_sparse_weight().item()  # must be scalar value

    # embed dictionary
    dict_sparse_embeds = biosyn.embed_sparse(names=eval_dictionary[:, 0], show_progress=True)
    dict_dense_embeds = biosyn.embed_dense(names=eval_dictionary[:, 0], show_progress=True)
    bert_encoder.eval()
    nested_encoder.eval()

    queries = []
    for i, nested_queries in enumerate(tqdm(nested_eval_queries, total=len(nested_eval_queries))):
        if isinstance(nested_queries, np.ndarray):
            nested_queries = nested_queries.tolist()
        nested_entity_padding_mask = torch.from_numpy(entity_padding_masks[i]).unsqueeze(0).to(device)
        query_encodings = bert_tokenizer(nested_queries, return_token_type_ids=True, padding="max_length",
                                         max_length=max_length, truncation=True, return_tensors="pt")
        input_ids = query_encodings["input_ids"].unsqueeze(0).to(device)
        attention_mask = query_encodings["attention_mask"].unsqueeze(0).to(device)
        token_type_ids = query_encodings["token_type_ids"].unsqueeze(0).to(device)
        assert input_ids.dim() == 3
        assert attention_mask.dim() == 3
        assert token_type_ids.dim() == 3
        (batch_size, nested_depth, max_length) = input_ids.size()
        with torch.no_grad():
            mention_sparse_embeds = biosyn.embed_sparse(np.array(nested_queries)).detach().cpu().numpy()
            query_embed = nested_encoder.encode_nested_queries(input_ids=input_ids, attention_mask=attention_mask,
                                                               token_type_ids=token_type_ids,
                                                               batch_size=batch_size,
                                                               nested_entities_mask=nested_entity_padding_mask,
                                                               nested_depth=nested_depth, max_length=max_length)
            query_embed = query_embed.view(batch_size * nested_depth, -1).detach().cpu().numpy()
            sparse_score_matrix = biosyn.get_score_matrix(
                query_embeds=mention_sparse_embeds,
                dict_embeds=dict_sparse_embeds
            )
            dense_score_matrix = biosyn.get_score_matrix(
                query_embeds=query_embed,
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
            for j, c_idx in enumerate(candidate_idxs):
                np_candidates = eval_dictionary[c_idx].squeeze()
                golden_cui = nested_entity_cuis[i][j]
                mention = nested_eval_queries[i][j]
                mask = entity_padding_masks[i, j]

                if mask == 0:
                    continue


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

        queries.append({
            'mentions': dict_mentions
        })

    result = {
        'queries': queries
    }

    return result





def evaluate_nested(nested_encoder, biosyn, eval_dictionary, nested_entity_mentions, nested_entity_cuis,
             entity_padding_masks, topk, score_mode, device):
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
    result = predict_topk(nested_encoder, biosyn, eval_dictionary, nested_entity_mentions,
                          nested_entity_cuis, entity_padding_masks, topk, score_mode, device)

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