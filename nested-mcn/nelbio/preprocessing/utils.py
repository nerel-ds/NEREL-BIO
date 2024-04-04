import codecs
import logging
import os
import re
from typing import Tuple, List, Dict, Union

ENTITY_ID_PATTERN = r"(?P<letter>[TRN])(?P<number>[0-9]+)"
ANNOTATION_FILENAME_PATTERNS = {
    "ru": r"(?P<file_id>[0-9]+)_ru.ann",
    "en": r"(?P<file_id>[0-9]+)_en.ann",
}
NOT_ANNOTATED_FILENAME_PATTERNS = {
    "ru": r"(?P<file_id>[0-9]+)_ru.txt",
    "en": r"(?P<file_id>[0-9]+)_en.txt",
}
CUI_PATTERN = r"C[0-9]+"


class Entity:
    def __init__(self, e_id, spans, e_type, entity_str):
        self.e_id = e_id
        self.spans = spans

        self.e_type = e_type
        self.entity_str = entity_str
        self.cui = None

    def __str__(self):
        min_span = min((int(t[0]) for t in self.spans))
        max_span = max((int(t[1]) for t in self.spans))

        return f"{self.e_id}||{min_span}|{max_span}||{self.e_type}||{self.entity_str}||{self.cui}"

    def to_biosyn_str(self):

        assert '||' not in self.e_id
        assert '||' not in self.e_type
        assert '||' not in self.entity_str

        return self.__str__()


class Relation:
    def __init__(self, head, rel, tail):
        self.head = head
        self.rel = rel
        self.tail = tail


def read_brat_annotation_file(input_path: str) -> Tuple[List[Entity], List[Relation]]:
    entity_id2entity: Dict[str, Entity] = {}

    relations: List[Relation] = []
    with codecs.open(input_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            line = line.replace("UMLS: C", "UMLS:C")
            attrs = line.strip().split('\t')

            entity_id = attrs[0]
            m = re.fullmatch(ENTITY_ID_PATTERN, entity_id)

            assert m is not None
            entity_letter = m.group("letter")
            # Processing mention spans
            if entity_letter == "T":
                assert len(attrs) == 3
                entity_type_and_span = attrs[1]
                e_t_s_split = entity_type_and_span.split()

                entity_type = e_t_s_split[0]
                spans = [t.split() for t in entity_type_and_span[len(entity_type):].strip().split(';')]

                mention_string = attrs[2]
                entity = Entity(e_id=entity_id, spans=spans, e_type=entity_type, entity_str=mention_string)
                entity_id2entity[entity_id] = entity

            # Processing the linking to dictionary
            elif entity_letter == "N":

                assert len(attrs) == 3 or len(attrs) == 2
                rel_type_and_entity_id_and_concept_id = attrs[1]

                rt_eid_cid_split = rel_type_and_entity_id_and_concept_id.split()

                assert len(rt_eid_cid_split) == 3
                rel_type = rt_eid_cid_split[0]
                assert rel_type == "Reference"
                entity_id = rt_eid_cid_split[1]

                vocab_concept_id = rt_eid_cid_split[2]

                assert vocab_concept_id.startswith("UMLS:")
                cui = vocab_concept_id.lstrip("UMLS:")
                assert entity_id2entity.get(entity_id) is not None
                entity_id2entity[entity_id].cui = cui

            elif entity_letter == "R":
                # R79	FINDING_OF Arg1:T103 Arg2:T70
                assert len(attrs) == 2
                rel_type_and_arguments = attrs[1]
                rel_args_list = rel_type_and_arguments.split()

                assert len(rel_args_list) == 3

                rel_type = rel_args_list[0]
                rel_arg_1 = rel_args_list[1]
                rel_arg_2 = rel_args_list[2]

                assert rel_arg_1[:5].lower() == "arg1:"
                assert rel_arg_2[:5].lower() == "arg2:"

                rel_arg_1 = rel_arg_1[5:]
                rel_arg_2 = rel_arg_2[5:]
                relation = Relation(head=rel_arg_1, rel=rel_type, tail=rel_arg_2)
                relations.append(relation)
    entities = list(entity_id2entity.values())

    return entities, relations


def read_brat_directory(directory: str, ann_filename_pattern: str, read_to_dict) \
        -> Dict[str, Dict[str, Union[List[Entity], List[Relation]]]]:
    logging.info(f"Processing directory: {directory}")
    for filename in os.listdir(directory):
        if not filename.endswith("ann"):
            continue
        m = re.fullmatch(ann_filename_pattern, filename)
        assert m is not None
        file_id = m.group("file_id")
        assert read_to_dict.get(file_id) is None

        file_path = os.path.join(directory, filename)
        entities, relations = read_brat_annotation_file(input_path=file_path)
        read_to_dict[file_id] = {
            "entities": entities,
            "relations": relations,
        }
    logging.info(f"Finished processing directory: {directory}")
    return read_to_dict


def read_brat_directories(dir_list: List[str], ann_filename_pattern: str) \
        -> Dict[str, Dict[str, Union[List[Entity], List[Relation]]]]:
    data_dict = {}
    for directory in dir_list:
        read_brat_directory(directory, ann_filename_pattern, read_to_dict=data_dict)
    return data_dict


def convert_brat_dirs_to_biosyn(dir_list: List[str], ann_filename_pattern: str, drop_cuiless: bool, output_dir: str):
    data_dict = read_brat_directories(dir_list, ann_filename_pattern)
    output_entities_dir = os.path.join(output_dir, "entities/")
    output_relations_dir = os.path.join(output_dir, "relations/")
    output_unique_mentions = os.path.join(output_dir, "unique_mentions.txt")
    output_unique_mentions_with_cui = os.path.join(output_dir, "unique_mentions_with_cui.txt")
    output_unique_mc_with_cui = os.path.join(output_dir, "unique_mc_with_cui.txt")

    output_unique_cuis = os.path.join(output_dir, "unique_cuis.txt")

    if not os.path.exists(output_entities_dir):
        os.makedirs(output_entities_dir)
    if not os.path.exists(output_relations_dir):
        os.makedirs(output_relations_dir)
    num_mentions = 0
    num_mentions_with_cui = 0
    unique_mentions = set()
    unique_mentions_with_cui = set()
    unique_mc_with_cui = set()
    unique_cuis = set()

    for file_id, entities_relations in data_dict.items():
        output_entities_path = os.path.join(output_entities_dir, f"{file_id}.concept")
        output_relations_path = os.path.join(output_relations_dir, f"{file_id}.relations")

        entities, relations = entities_relations["entities"], entities_relations["relations"]
        num_mentions += len(entities)
        num_mentions_with_cui += sum((1 for e in entities if e.cui is not None and e.cui != "NULL"))
        for e in entities:
            if e.cui is not None and e.cui != '' and e.cui != "NULL":
                unique_mentions_with_cui.add(e.entity_str)
                unique_mc_with_cui.add(f"{e.cui}|{e.entity_str}")
            if e.cui == '':
                print(e.entity_str)

        unique_mentions.update((e.entity_str for e in entities))
        unique_cuis.update((e.cui for e in entities))

        write_entities_biosyn(output_entities_path, entities, drop_cuiless)
        write_relations_biosyn(output_relations_path, relations)
    logging.info(f"Overall number of entities: {num_mentions}")
    logging.info(f"Number of entities with a CUI: {num_mentions_with_cui}")
    logging.info(f"Unique unique_mentions_with_cui: {len(unique_mentions_with_cui)}")
    logging.info(f"Unique unique_mc_with_cui: {len(unique_mc_with_cui)}")
    logging.info(f"Unique number of mentions: {len(unique_mentions)}")
    logging.info(f"Unique number of CUIS: {len(unique_cuis)}")
    with open(output_unique_mentions, 'w', encoding="utf-8") as out_file:
        for m in unique_mentions:
            out_file.write(f"{m.strip()}\n")
    with open(output_unique_cuis, 'w', encoding="utf-8") as out_file:
        for c in unique_cuis:
            out_file.write(f"{c}\n")
    with open(output_unique_mentions_with_cui, 'w', encoding="utf-8") as out_file:
        for m in unique_mentions_with_cui:
            out_file.write(f"{m}\n")
    with open(output_unique_mc_with_cui, 'w', encoding="utf-8") as out_file:
        for mc in unique_mc_with_cui:
            out_file.write(f"{mc}\n")


def write_entities_biosyn(output_path: str, entities: List[Entity], drop_cuiless):
    with codecs.open(output_path, 'w+', encoding="utf-8") as out_file:
        for entity in entities:
            cui = entity.cui
            if not drop_cuiless or (isinstance(cui, str) and re.fullmatch(CUI_PATTERN, cui) is not None):
                out_file.write(f"{entity.to_biosyn_str()}\n")


def write_relations_biosyn(output_path: str, relations: List[Relation]):
    with codecs.open(output_path, 'w+', encoding="utf-8") as out_file:
        for relation in relations:
            out_file.write(f"{relation.head}\t{relation.rel}\t{relation.tail}\n")


def save_entities_as_biosyn(entities, output_dir: str):
    output_path = os.path.join(output_dir, "0.concept")
    with open(output_path, 'w', encoding="utf-8") as out_file:
        for (flat_q, sep_context, flat_cui, nested_mentions_list) in entities:
            for nm in nested_mentions_list:
                assert "|" not in nm
            out_file.write(f"{'|'.join(nested_mentions_list)}||-1|-1||{sep_context}||{flat_q}||{flat_cui}\n")


def save_data_split_biosyn(data_split, output_dir):
    train_entities, dev_entities, test_entities = data_split
    output_train_dir = os.path.join(output_dir, "train/")
    output_dev_dir = os.path.join(output_dir, "dev/")
    output_test_dir = os.path.join(output_dir, "test/")
    if not os.path.exists(output_train_dir):
        os.makedirs(output_train_dir)
    if not os.path.exists(output_dev_dir):
        os.makedirs(output_dev_dir)
    if not os.path.exists(output_test_dir):
        os.makedirs(output_test_dir)
    save_entities_as_biosyn(train_entities, output_train_dir)
    save_entities_as_biosyn(dev_entities, output_dev_dir)
    save_entities_as_biosyn(test_entities, output_test_dir)
