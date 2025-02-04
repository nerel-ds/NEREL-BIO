from typing import Dict, List, Tuple, Set

import pandas as pd

from bionnel.utils.entity import Entity

# LANG_COL = "lang"
DOC_ID_COL = "document_id"
SPANS_COL = "spans"
TEXT_COL = "text"
ENT_TYPE_COL = "entity_type"
CUI_COL = "UMLS_CUI"


def entities_groupby_nested(df: pd.DataFrame, spans_sep: str = ',', start_end_sep='-'):
    document_id2entities: Dict[str, List[Entity]] = {}
    # Grouping entities by document_id
    for _, row in df.iterrows():
        doc_id = row[DOC_ID_COL]
        spans_s = row[SPANS_COL]
        spans = [tuple(map(int, s.split(start_end_sep))) for s in spans_s.strip().split(spans_sep)]
        entity_str = row[TEXT_COL]
        e_type = row[ENT_TYPE_COL]
        cui = row[CUI_COL]

        entity = Entity(doc_id=doc_id, e_id=None, spans=spans, e_type=e_type, entity_str=entity_str, cui=cui)
        if document_id2entities.get(doc_id) is None:
            document_id2entities[doc_id] = []
        document_id2entities[doc_id].append(entity)

    return document_id2entities


def filter_unnested_entities(nested_entities: List[Entity]) -> List[Entity]:
    keep_entities: List[Entity] = []
    nested_entities = sorted(nested_entities, key=lambda e: len(e.entity_str))
    longest_entity = nested_entities[-1]
    longest_entity_str = longest_entity.entity_str.lower()
    for ent in nested_entities:
        ent_str = ent.entity_str.lower()
        if ent_str in longest_entity_str:
            keep_entities.append(ent)
        else:
            pass
            # print(f"`{ent_str}` is not in `{longest_entity_str}`")
    return keep_entities


def entities2nested_entities_list(entities: List[Entity], drop_non_nested=False) -> List[List[Entity]]:
    entity_spans_with_start_end_type: List[Tuple[int, Entity, str]] = []
    for ent in entities:
        span_start = ent.min_span
        span_end = ent.max_span
        entity_spans_with_start_end_type.append((span_start, ent, "<s>"))
        entity_spans_with_start_end_type.append((span_end, ent, "<e>"))

    entity_spans_with_start_end_type.sort(key=lambda t: t[0])
    opened_entities: Set[Entity] = set()
    closed_entities: Set[Entity] = set()

    # entity_id2nested_entities_list: Dict[str, List[str]] = {}
    nested_entities_list: List[List[Entity]] = []
    for (span_position, entity, span_type) in entity_spans_with_start_end_type:

        assert span_type in ("<s>", "<e>")
        if span_type == "<s>":
            opened_entities.add(entity)
        elif span_type == "<e>":
            closed_entities.add(entity)
        else:
            raise Exception(f"Invalid span type: {span_type}")
        if len(opened_entities) == len(closed_entities):
            # print("AAAAA")
            opened_ent_strs = set((str(e) for e in opened_entities))
            closed_ent_strs = set((str(e) for e in closed_entities))
            assert len(opened_ent_strs.intersection(closed_ent_strs)) == len(closed_ent_strs)
            if drop_non_nested:
                assert len(opened_entities) != 0
                if len(opened_entities) > 1:
                    nested_entities_list.append(list(opened_entities))

            else:
                nested_entities_list.append(list(opened_entities))

            opened_entities.clear()
            closed_entities.clear()
    nested_entities_list = [filter_unnested_entities(ent_list) for ent_list in nested_entities_list]

    return nested_entities_list


def create_nestedness_lists(document_id2entities: Dict[str, List[Entity]]) -> List[List[Entity]]:
    nested_entities: List[List[Entity]] = []
    for doc_ic, entities in document_id2entities.items():
        # print(len(entities2nested_entities_list(entities=entities)))
        nested_entities.extend(entities2nested_entities_list(entities=entities))
    return nested_entities
