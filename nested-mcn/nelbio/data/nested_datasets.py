import glob
import itertools
import logging
import os
from abc import ABC
from typing import Dict, List, Tuple, Set

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from nelbio.utils.nested_utils import check_query_label

LOGGER = logging.getLogger(__name__)


class NestedQueryDataset(Dataset):

    def __init__(self, data_dir,
                 pad_nested=True,
                 filter_composite=False,
                 filter_cuiless=False,
                 cuis_vocab=None,
                 drop_not_nested=False,

                 ):
        """
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries
        draft : bool
            use subset of queries for debugging (default False)
        """
        LOGGER.info("QueryDataset! data_dir={}\n"
                    "filter_composite={} filter_cuiless={} drop_not_nested={}".format(
            data_dir, filter_composite, filter_cuiless, drop_not_nested
        ))

        nested_entity_mentions, nested_entity_cuis, nested_contribution_masks, entity_padding_masks, max_nesting_depth = self.load_data(
            data_dir=data_dir,
            cuis_vocab=cuis_vocab,
            drop_not_nested=drop_not_nested,
            filter_composite=filter_composite,
            filter_cuiless=filter_cuiless,
            pad_nested=pad_nested
        )
        self.nested_entity_mentions = nested_entity_mentions
        self.nested_entity_cuis = nested_entity_cuis
        self.nested_contribution_masks = nested_contribution_masks
        self.entity_padding_masks = entity_padding_masks
        self.max_nesting_depth = max_nesting_depth

    @staticmethod
    def update_nested_entities(entity_spans_with_start_end_type: List[Tuple[int, str, str]],
                               nested_entity_mentions: List[List[str]],
                               nested_entity_cuis: List[List[str]],
                               entity_id2mention: Dict[str, str],
                               entity_id2cui: Dict[str, str],
                               drop_not_nested=False):

        entity_spans_with_start_end_type.sort(key=lambda t: t[0])
        opened_entity_ids: Set[str] = set()
        closed_entity_ids: Set[str] = set()
        # entity_id2nested_entities_list: Dict[str, List[str]] = {}
        # nested_entities_list: List[List[str]] = []
        for (span_position, entity_id, span_type) in entity_spans_with_start_end_type:

            assert span_type in ("<s>", "<e>")
            if span_type == "<s>":
                opened_entity_ids.add(entity_id)
            elif span_type == "<e>":
                closed_entity_ids.add(entity_id)
            else:
                raise Exception(f"Invalid span type: {span_type}")
            if len(opened_entity_ids) == len(closed_entity_ids):
                assert len(opened_entity_ids.intersection(closed_entity_ids)) == len(closed_entity_ids)
                if drop_not_nested:
                    assert len(opened_entity_ids) != 0
                    if len(opened_entity_ids) > 1:
                        nested_entity_mentions.append([entity_id2mention[e_id] for e_id in opened_entity_ids])
                        nested_entity_cuis.append([entity_id2cui[e_id] for e_id in opened_entity_ids])
                else:
                    nested_entity_mentions.append([entity_id2mention[e_id] for e_id in opened_entity_ids])
                    nested_entity_cuis.append([entity_id2cui[e_id] for e_id in opened_entity_ids])

                opened_entity_ids.clear()
                closed_entity_ids.clear()

    def load_data(self, data_dir, cuis_vocab, drop_not_nested,
                  filter_composite, filter_cuiless, pad_nested=True):
        """
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries
        filter_cuiless : bool
            remove samples with cuiless
        Returns
        -------
        data : np.array
            mention, cui pairs
        """
        entity_ids = []

        concept_files = glob.glob(os.path.join(data_dir, "*.concept"))
        entity_spans: List[Tuple[int, str, str]] = []

        concept_entity_id2mention: Dict[str, str] = {}
        concept_entity_id2cui: Dict[str, str] = {}

        nested_entity_mentions: List[List[str]] = []
        nested_entity_cuis: List[List[str]] = []

        masked_counter = 0
        nested_deleted_counter = 0
        composite_counter = 0

        for concept_file in tqdm(concept_files):
            file_id = concept_file.split('/')[-1].split('.')[0]

            with open(concept_file, "r", encoding='utf-8') as f:
                concepts = f.readlines()
            entity_spans.clear()

            for concept in concepts:
                concept = concept.split("||")
                entity_id = f"{file_id}_{concept[0].strip()}"
                spans = concept[1].strip()
                span_start = int(spans.split('|')[0])
                span_end = int(spans.split('|')[1])

                mention = concept[3].strip()

                cui = concept[4].strip()
                is_composite = (cui.replace("+", "|").count("|") > 0)

                # filter composite cui
                if filter_composite and is_composite:
                    composite_counter += 1
                    continue
                # filter cuiless
                # print(cui)
                # print(cuis_vocab)
                # print('--')
                if cui in ('NULL', "None"):
                    cui = "-D"
                if cuis_vocab is not None:
                    if cui not in cuis_vocab:
                        cui = "-D"

                entity_spans.append((span_start, entity_id, "<s>"))
                entity_spans.append((span_end, entity_id, "<e>"))
                concept_entity_id2mention[entity_id] = mention
                concept_entity_id2cui[entity_id] = cui

                # mentions_with_cuis.append((mention, cui))
                entity_ids.append(entity_id)
            self.update_nested_entities(entity_spans_with_start_end_type=entity_spans,
                                        nested_entity_mentions=nested_entity_mentions,
                                        nested_entity_cuis=nested_entity_cuis,
                                        entity_id2mention=concept_entity_id2mention,
                                        entity_id2cui=concept_entity_id2cui,
                                        drop_not_nested=drop_not_nested)

        assert len(nested_entity_mentions) == len(nested_entity_cuis)
        max_nesting_depth = max((len(t) for t in nested_entity_mentions))

        LOGGER.info(f"Finished loading nested entities. Maximum nesting depth is {max_nesting_depth}")
        nested_entity_mentions, nested_entity_cuis, nested_contribution_mask, c1, c2 = self.filter_nested_entities(
            nested_mentions=nested_entity_mentions,
            nested_cuis=nested_entity_cuis
        )
        masked_counter += c1
        nested_deleted_counter += c2
        num_nested = len(nested_entity_mentions)
        num_flattened = sum(len(t) for t in nested_entity_mentions)

        entity_padding_masks = None
        if pad_nested:
            LOGGER.info(f"Padding entities...")
            nested_entity_mentions, entity_padding_masks = self.pad_nested_entities(
                nested_entities=nested_entity_mentions,
                max_nesting_depth=max_nesting_depth)
            nested_entity_cuis, entity_cui_padding_masks = self.pad_nested_entities(nested_entities=nested_entity_cuis,
                                                                                    max_nesting_depth=max_nesting_depth)
            assert np.array_equal(entity_padding_masks, entity_cui_padding_masks)
            del entity_cui_padding_masks
            LOGGER.info(f"Finished entities padding.")

        LOGGER.info(f"Masked {masked_counter} entities in nested entities.\n"
                    f"Completely removed {nested_deleted_counter} nested entities.\n"
                    f"{num_nested} nested entities remaining ({num_flattened} flattened  entities).\n"
                    f"Removed {composite_counter} composite cuis.")

        return nested_entity_mentions, nested_entity_cuis, nested_contribution_mask, entity_padding_masks, max_nesting_depth

    def filter_nested_entities(self, nested_mentions: List[List[str]],
                               nested_cuis: List[List[str]], ) \
            -> Tuple[List[List[str]], List[List[str]], List[List[int]], int, int]:
        """
        Creates padded nested entities filtration mask: mentions that have not been marked to be deleted
        (Entities with CUI "-D" are deleted mentions) will have value 1 in filtration mask and 0 otherwise.
        Normally, deleted mentions are not removed. They are marked to be removed from loss calculation but will
        contribute to normalization of non-deleted entities.
        However, nested entities that consist of deleted mentions only, will be really removed.
        """

        filtered_n_m = []
        filtered_n_cuis = []
        filtered_n_contribution_mask = []
        masked_c = 0
        n_deleted_c = 0
        for ms, cuis in zip(nested_mentions, nested_cuis):
            contribution_mask = []
            remove_nested_entity = True
            for c in cuis:
                if c == "-D":
                    contribution_mask.append(0)
                    masked_c += 1
                else:
                    contribution_mask.append(1)
                    remove_nested_entity = False
            if remove_nested_entity:
                masked_c -= len(cuis)
                n_deleted_c += 1
            else:
                filtered_n_m.append(ms)
                filtered_n_cuis.append(cuis)
                filtered_n_contribution_mask.append(contribution_mask)
        return filtered_n_m, filtered_n_cuis, filtered_n_contribution_mask, masked_c, n_deleted_c

    def pad_nested_entities(self, nested_entities: List[List[str]], max_nesting_depth: int) \
            -> Tuple[np.ndarray, np.ndarray]:
        padded_entities = []
        padding_masks = []
        for nested_entity in nested_entities:
            real_length = len(nested_entity)
            padding_length = max_nesting_depth - real_length
            assert padding_length >= 0
            padded_entity = nested_entity + ["", ] * padding_length
            padding_mask = [1, ] * real_length + [0, ] * padding_length

            padded_entities.append(padded_entity)
            padding_masks.append(padding_mask)
        padded_entities = np.array(padded_entities)
        padding_masks = np.array(padding_masks)

        return padded_entities, padding_masks


class AbstractNestedDataset(ABC):
    nested_entity_cuis: List[List[str]]
    topk: int
    dict_ids: np.array

    def get_labels(self, query_idx, candidate_idx):
        nested_cuis = self.nested_entity_cuis[query_idx]
        (nested_depth, topk) = candidate_idx.shape
        assert topk == self.topk
        # <depth, topk>
        nested_candidate_cuis = self.dict_ids[candidate_idx.reshape((-1,))].reshape((nested_depth, topk))

        assert len(nested_cuis) == nested_depth

        labels = np.zeros(shape=(nested_depth, self.topk))

        for i in range(nested_depth):
            query_cui = nested_cuis[i]
            for j in range(topk):
                candidate_cui = nested_candidate_cuis[i, j]
                label = check_query_label(query_cui, candidate_cui)
                labels[i, j] = label

        return labels
