import logging
import random
from typing import List, Union

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

LOGGER = logging.getLogger(__name__)


class NestedSepFlatCandidateDataset(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """

    def __init__(self, sep_contexts, flat_queries, flat_cuis, n_m_list, dictionary, tokenizer, query_max_length,
                 context_max_length, topk, d_ratio, s_score_matrix, s_candidate_idxs):

        """
        Retrieve top-k candidates based on sparse/dense embedding
        Parameters
        ----------
        queries : list
            A list of tuples (name, id)
        dictionary : list
            A list of tuples (name, id)
        tokenizer : BertTokenizer
            A BERT tokenizer for dense embedding
        topk : int
            The number of candidates
        d_ratio : float
            The ratio of dense candidates from top-k
        s_score_matrix : np.array
        s_candidate_idxs : np.array
        """
        # assert nested_queries.shape == nested_entity_cuis.shape == entity_padding_masks.shape
        LOGGER.info(f"CandidateDataset! "
                    f"len(dicts)={len(dictionary)}, topk={topk}, d_ratio={d_ratio}")
        self.context_sep_token = tokenizer.sep_token

        self.sep_contexts = sep_contexts
        self.flat_queries = flat_queries
        self.flat_cuis = flat_cuis
        # self.flat_in_n_pos_ids = flat_in_n_pos_ids
        self.n_m_list = n_m_list
        assert len(flat_cuis) == len(flat_queries)
        assert len(sep_contexts) == len(n_m_list)


        self.dict_names = [row[0] for row in dictionary]
        self.dict_ids = np.array([row[1] for row in dictionary])
        self.topk = topk
        self.n_dense = int(topk * d_ratio)
        self.n_sparse = topk - self.n_dense
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.context_max_length = context_max_length

        self.s_score_matrix = s_score_matrix
        self.s_candidate_idxs = s_candidate_idxs
        self.d_candidate_idxs = None

    @staticmethod
    def create_flat_entities_with_sep_context(nested_queries: List[List[str]], nested_entity_cuis: List[List[str]],
                                              nested_contribution_masks: List[List[int]], context_sep_token,
                                              keep_longest_entity_only):

        sep_contexts: List[Union[str, List[str]]] = []
        # flat_id2context_id: List[int] = []
        flat_queries: List[str] = []
        flat_cuis: List[str] = []
        flat_counter = 0
        nested_counter = 0
        # flat_in_n_pos_ids = []
        n_m_list: List[List[str]] = []
        assert len(nested_queries) == len(nested_entity_cuis)
        for nested_e_id, (queries, cuis, masks) in \
                enumerate(zip(nested_queries, nested_entity_cuis, nested_contribution_masks)):
            assert len(queries) == len(cuis)

            nested_counter += 1
            for i, (q, c, m) in enumerate(zip(queries, cuis, masks)):
                assert (c == "-D" and m == 0) or (c != "-D" and m == 1)
                if c != "-D":
                    # flat_id2context_id.append(nested_e_id)
                    flat_queries.append(q)
                    max_query_length = 0
                    if keep_longest_entity_only and len(queries) != 1:
                        max_query_length = max((len(ent) for ent in queries if ent != q))
                    query_nested_neighboring_mentions = [q, ] + [qq for qq in queries if
                                                                 q != qq and len(qq) >= max_query_length]
                    query_sep_mentions = context_sep_token.join(query_nested_neighboring_mentions)

                    flat_cuis.append(c)
                    flat_counter += 1
                    # flat_in_n_pos_ids.append(i)
                    sep_contexts.append(query_sep_mentions)
                    n_m_list.append(query_nested_neighboring_mentions)

        return sep_contexts, flat_queries, flat_cuis, n_m_list, nested_counter

    def set_dense_candidate_idxs(self, d_candidate_idxs):
        self.d_candidate_idxs = d_candidate_idxs

    def set_s_score_matrix(self, s_score_matrix):
        self.s_score_matrix = s_score_matrix

    def set_s_candidate_idxs(self, s_candidate_idxs):
        self.s_candidate_idxs = s_candidate_idxs

    def check_label(self, query_id, candidate_id_set):
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

    def get_labels(self, query_idx, candidate_idxs):
        labels = np.array([])
        query_id = self.flat_cuis[query_idx]
        candidate_ids = np.array(self.dict_ids)[candidate_idxs]
        for candidate_id in candidate_ids:
            label = self.check_label(query_id, candidate_id)
            labels = np.append(labels, label)
        return labels

    def __getitem__(self, query_idx):
        assert (self.s_candidate_idxs is not None)
        assert (self.s_score_matrix is not None)
        assert (self.d_candidate_idxs is not None)

        query = self.flat_queries[query_idx]
        # e1 [sep] e2 [sep] .. en [sep]
        sep_context = self.sep_contexts[query_idx]

        # flat_in_nested_id = self.flat_in_n_pos_ids[query_idx]
        nested_mentions: List[str] = self.n_m_list[query_idx]
        num_nested_entities = len(nested_mentions)

        query_token = self.tokenizer(query, max_length=self.query_max_length, return_token_type_ids=True,
                                     padding='max_length', truncation=True, return_tensors='pt')
        sep_context_token = self.tokenizer(sep_context, max_length=self.context_max_length, return_token_type_ids=True,
                                           padding='max_length', truncation=True, return_tensors='pt')
        # combine sparse and dense candidates as many as top-k
        s_candidate_idx = self.s_candidate_idxs[query_idx]

        # assert len(s_candidate_idx.shape) == 2

        d_candidate_idx = self.d_candidate_idxs[query_idx]
        # assert len(d_candidate_idx.shape) == 2

        # fill with sparse candidates first
        topk_candidate_idx = s_candidate_idx[:self.n_sparse]

        # fill remaining candidates with dense
        for d_idx in d_candidate_idx:
            if len(topk_candidate_idx) >= self.topk:
                break
            if d_idx not in topk_candidate_idx:
                topk_candidate_idx = np.append(topk_candidate_idx, d_idx)

        # sanity check
        assert len(topk_candidate_idx) == self.topk
        assert len(topk_candidate_idx) == len(set(topk_candidate_idx))

        candidate_names = [self.dict_names[candidate_idx] for candidate_idx in topk_candidate_idx]
        nested_sep_candidates_inp_ids = []
        nested_sep_candidates_att_masks = []
        for name in candidate_names:
            nested_mentions_copy = nested_mentions.copy()
            nested_mentions_copy[0] = name
            nested_candidate_s = self.context_sep_token.join(nested_mentions_copy)
            nested_candidate_token_out = self.tokenizer(nested_candidate_s, max_length=self.context_max_length,
                                                        padding='max_length', truncation=True, return_tensors='pt')
            nested_candidate_inp_ids = nested_candidate_token_out["input_ids"][0]
            nested_candidate_att_mask = nested_candidate_token_out["attention_mask"][0]

            nested_sep_candidates_inp_ids.append(nested_candidate_inp_ids)
            nested_sep_candidates_att_masks.append(nested_candidate_att_mask)

        nested_sep_candidates_inp_ids = torch.stack(nested_sep_candidates_inp_ids)
        nested_sep_candidates_att_masks = torch.stack(nested_sep_candidates_att_masks)
        # nested_sep_candidates_input = (nested_sep_candidates_inp_ids, nested_sep_candidates_att_masks)
        nested_sep_candidates_input = {
            "input_ids": nested_sep_candidates_inp_ids,
            "attention_mask": nested_sep_candidates_att_masks
        }

        candidate_s_scores = self.s_score_matrix[query_idx][topk_candidate_idx]
        labels = self.get_labels(query_idx, topk_candidate_idx).astype(np.float32)

        candidate_token = self.tokenizer(candidate_names, max_length=self.query_max_length, return_token_type_ids=True,
                                         padding='max_length', truncation=True, return_tensors='pt')
        nested_mask = 1 if num_nested_entities > 1 else 0
        # print(nested_mask, num_nested_entities, nested_mentions)
        d = {
            "query_token": query_token,
            "context_token": sep_context_token,
            "candidate_token": candidate_token,
            "nested_mask": nested_mask,
            "candidate_s_scores": candidate_s_scores,
            "nested_sep_candidate_input": nested_sep_candidates_input,
            "labels": labels
        }

        return d
        # return (nested_query_token, candidate_token, candidate_s_scores, contribution_mask), nested_depth, labels

    def collate_fn(self, batch):
        batch_size = len(batch)
        max_child_num = 0
        query_token = []
        sep_context_token = []
        candidate_token = []
        candidate_s_scores = []
        nested_sep_candidate_input = []
        nested_mask = []
        labels = []

        for d in batch:
            query_token.append(d["query_token"])
            sep_context_token.append(d["context_token"])
            candidate_token.append(d["candidate_token"])
            candidate_s_scores.append(d["candidate_s_scores"])
            nested_sep_candidate_input.append(d["nested_sep_candidate_input"])
            nested_mask.append(d["nested_mask"])
            labels.append(d["labels"])

            max_child_num = max(max_child_num, len(d["context_token"]["input_ids"]))

        batch_query_token = default_collate(query_token)
        batch_sep_context_token = default_collate(sep_context_token)
        batch_candidate_token = default_collate(candidate_token)
        batch_candidate_s_scores = default_collate(candidate_s_scores)
        context_max_length = sep_context_token[0]["input_ids"].size()[-1]
        assert context_max_length == self.context_max_length
        batch_labels = default_collate(labels)

        batch_nested_sep_cand_input_ids = torch.zeros(size=(batch_size, self.topk, context_max_length),
                                                      dtype=torch.int64)
        batch_nested_sep_cand_att_masks = torch.zeros(size=(batch_size, self.topk, context_max_length),
                                                      dtype=torch.float32)

        for i, (cont_t, n_s_c_i) in enumerate(zip(sep_context_token, nested_sep_candidate_input)):
            # <topk, context_max_length>
            nested_sep_c_inp_ids = n_s_c_i["input_ids"]
            nested_sep_c_att_masks = n_s_c_i["attention_mask"]
            assert nested_sep_c_inp_ids.size() == nested_sep_c_att_masks.size()

            batch_nested_sep_cand_input_ids[i, :] = nested_sep_c_inp_ids
            batch_nested_sep_cand_att_masks[i, :] = nested_sep_c_att_masks

        nested_sep_candidate_input = {
            "input_ids": batch_nested_sep_cand_input_ids,
            "attention_mask": batch_nested_sep_cand_att_masks,
        }
        nested_mask = torch.FloatTensor(nested_mask)
        d = {
            "query_token": batch_query_token,
            "context_input": batch_sep_context_token,
            "candidate_token": batch_candidate_token,
            "nested_mask": nested_mask,
            "nested_sep_candidate_input": nested_sep_candidate_input,
            "candidate_s_scores": batch_candidate_s_scores,
            "labels": batch_labels
        }
        return d

    def __len__(self):
        return len(self.flat_queries)


class NestedSepEvaluationQueryDatasetV2(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """

    def __init__(self, sep_contexts, flat_queries, flat_cuis, n_m_list,
                 tokenizer, query_max_length, context_max_length):

        self.sep_contexts = sep_contexts
        self.flat_queries = flat_queries
        self.flat_cuis = flat_cuis

        self.n_m_list = n_m_list
        assert len(flat_cuis) == len(flat_queries)
        assert len(sep_contexts) == len(n_m_list)
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.context_max_length = context_max_length

    def __len__(self):
        return len(self.flat_queries)

    def __getitem__(self, query_idx):
        query = self.flat_queries[query_idx]
        # e1 [sep] e2 [sep] .. en [sep]
        sep_context = self.sep_contexts[query_idx]
        # flat_in_nested_id = self.flat_in_n_pos_ids[query_idx]
        nested_mentions: List[str] = self.n_m_list[query_idx]
        cui = self.flat_cuis[query_idx]

        query_token = self.tokenizer(query, max_length=self.query_max_length, return_token_type_ids=False,
                                     padding='max_length', truncation=True, return_tensors='pt')
        sep_context_token = self.tokenizer(sep_context, max_length=self.context_max_length, return_token_type_ids=False,
                                           padding='max_length', truncation=True, return_tensors='pt')

        d = {
            "query_token": query_token,
            "query": query,
            "nested_mentions": nested_mentions,
            "sep_context_token": sep_context_token,
            "labels": cui
        }

        return d

    @staticmethod
    def collate_fn(samples):
        query_token = []
        query = []
        nested_mentions = []
        sep_context_token = []
        labels = []
        for d in samples:
            query_token.append(d["query_token"])
            query.append(d["query"])
            nested_mentions.append(d["nested_mentions"])
            sep_context_token.append(d["sep_context_token"])
            labels.append(d["labels"])

        batch_query_token = default_collate(query_token)
        sep_context_token = default_collate(sep_context_token)
        batch_query = default_collate(query)
        batch_labels = default_collate(labels)

        return batch_query_token, batch_query, nested_mentions, sep_context_token, batch_labels
