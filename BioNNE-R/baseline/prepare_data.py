#!/usr/bin/env python3
"""
Convert BioNNE-R TSV + text files to OpenNRE JSON lines format.

Supports three modes:
1. Labeled mode (relation TSV → JSON lines)
2. Labeled + negative sampling (relation TSV + entity TSV → JSON lines with no_relation)
3. Blind mode (entity TSV → all candidate pairs as JSON lines)

Usage:
    # Labeled (current behavior)
    python prepare_data.py eng-train-rel.tsv texts/ -o data/train.txt

    # Labeled + negative sampling
    python prepare_data.py eng-train-rel.tsv texts/ -o data/train.txt --entities ent.tsv --neg-ratio 3

    # Blind prediction (entity TSV → all candidate pairs)
    python prepare_data.py eng-test-ent.tsv texts/ -o data/test.txt
"""

import argparse
import json
import random
import re
from pathlib import Path

import nltk
import pandas as pd

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

RELATION_TYPES = [
    "ABBREVIATION", "ALTERNATIVE_NAME", "SUBCLASS_OF", "PART_OF",
    "TREATED_USING", "ORIGINS_FROM", "TO_DETECT_OR_STUDY", "AFFECTS",
    "HAS_CAUSE", "APPLIED_TO", "USED_IN", "ASSOCIATED_WITH",
    "PHYSIOLOGY_OF", "FINDING_OF",
    "no_relation",
]


def parse_config(config_path: str) -> set[tuple[str, str]]:
    """Parse annotation config and return set of valid (arg1_type, arg2_type) tuples.

    Adapted from clean_annotations.py parse_config().
    """
    valid_pairs = set()

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    relations_match = re.search(r"\[relations\](.*?)(?=\[|\Z)", content, re.DOTALL)
    if not relations_match:
        return valid_pairs

    relations_section = relations_match.group(1)
    for line in relations_section.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("<"):
            continue

        parts = line.split(None, 1)
        if len(parts) < 2:
            continue

        args_part = parts[1]

        # Extract Arg1 types
        arg1_match = re.search(r"Arg1:([A-Z_|]+)", args_part)
        if not arg1_match:
            continue
        arg1_types = arg1_match.group(1).split("|")

        # Extract Arg2 types
        arg2_match = re.search(r"Arg2:([A-Z_|]+)", args_part)
        if arg2_match:
            arg2_types = arg2_match.group(1).split("|")
        else:
            comma_match = re.search(r",\s*([A-Z_|]+)\s*$", args_part)
            if comma_match:
                arg2_types = comma_match.group(1).split("|")
            else:
                continue

        for t1 in arg1_types:
            for t2 in arg2_types:
                valid_pairs.add((t1, t2))

    return valid_pairs


def load_entities(entity_tsv: Path) -> dict[str, list[tuple[str, str, str]]]:
    """Read entity TSV → {doc_id: [(type, text, span), ...]}.

    Entity TSV columns: document_id, entity_type, entity_text, entity_span
    """
    df = pd.read_csv(entity_tsv, sep="\t")
    entities_by_doc: dict[str, list[tuple[str, str, str]]] = {}
    for _, row in df.iterrows():
        doc_id = str(row["document_id"])
        entry = (row["entity_type"], str(row["entity_text"]), str(row["entity_span"]))
        entities_by_doc.setdefault(doc_id, []).append(entry)
    return entities_by_doc


def generate_pairs(
    entities: list[tuple[str, str, str]],
    valid_type_pairs: set[tuple[str, str]] | None = None,
) -> list[tuple[tuple[str, str, str], tuple[str, str, str]]]:
    """Generate ordered entity pairs, optionally filtered by valid type combinations.

    Each entity is (type, text, span). Returns list of (head, tail) pairs.
    """
    pairs = []
    for i, e1 in enumerate(entities):
        for j, e2 in enumerate(entities):
            if i == j:
                continue
            if valid_type_pairs is not None and (e1[0], e2[0]) not in valid_type_pairs:
                continue
            pairs.append((e1, e2))
    return pairs


def load_texts(texts_dir: Path) -> dict[str, str]:
    """Load all .txt files from directory into {doc_id: text} dict."""
    texts = {}
    for txt_file in texts_dir.glob("*.txt"):
        texts[txt_file.stem] = txt_file.read_text(encoding="utf-8")
    return texts


def parse_span(span_str: str) -> tuple[int, int]:
    """Parse 'start-end' to (start, end)."""
    start, end = span_str.split("-")
    return int(start), int(end)


def find_sentence_segment(
    text: str,
    head_start: int,
    head_end: int,
    tail_start: int,
    tail_end: int,
    lang: str = "english",
) -> tuple[str, int]:
    """Find minimal sentence segment containing both entities.

    Returns (segment_text, offset).
    """
    try:
        sent_tokenizer = nltk.data.load(f"tokenizers/punkt_tab/{lang}.pickle")
    except LookupError:
        sent_tokenizer = nltk.data.load("tokenizers/punkt_tab/english.pickle")
    sentences = list(sent_tokenizer.span_tokenize(text))

    if not sentences:
        return text, 0

    entity_min = min(head_start, tail_start)
    entity_max = max(head_end, tail_end)

    first_idx = 0
    last_idx = len(sentences) - 1

    for i, (s_start, s_end) in enumerate(sentences):
        if s_start <= entity_min < s_end:
            first_idx = i
        if s_start < entity_max <= s_end:
            last_idx = i

    if first_idx > last_idx:
        first_idx, last_idx = last_idx, first_idx

    # Add one context sentence on each side
    first_idx = max(0, first_idx - 1)
    last_idx = min(len(sentences) - 1, last_idx + 1)

    seg_start = sentences[first_idx][0]
    seg_end = sentences[last_idx][1]
    return text[seg_start:seg_end], seg_start


def _make_instance(
    text: str,
    head_type: str,
    head_text: str,
    head_span: str,
    tail_type: str,
    tail_text: str,
    tail_span: str,
    relation: str,
    doc_id: str,
    lang: str,
) -> dict | None:
    """Build one OpenNRE JSON instance from entity pair info.

    Returns None if spans don't fit within the text.
    """
    head_start, head_end = parse_span(head_span)
    tail_start, tail_end = parse_span(tail_span)

    segment, offset = find_sentence_segment(
        text, head_start, head_end, tail_start, tail_end, lang
    )

    h_start = head_start - offset
    h_end = head_end - offset
    t_start = tail_start - offset
    t_end = tail_end - offset

    # Validate spans fit within segment
    if h_start < 0 or h_end > len(segment) or t_start < 0 or t_end > len(segment):
        # Fallback to full text
        segment = text
        h_start, h_end = head_start, head_end
        t_start, t_end = tail_start, tail_end

    return {
        "text": segment,
        "h": {"name": head_text, "pos": [h_start, h_end]},
        "t": {"name": tail_text, "pos": [t_start, t_end]},
        "relation": relation,
        "doc_id": doc_id,
        "head_span": head_span,
        "tail_span": tail_span,
        "head_type": head_type,
        "tail_type": tail_type,
    }


def add_negatives(
    positives_by_doc: dict[str, set[tuple[str, str]]],
    entities_by_doc: dict[str, list[tuple[str, str, str]]],
    neg_ratio: int,
    valid_type_pairs: set[tuple[str, str]] | None,
    seed: int,
) -> dict[str, list[tuple[tuple[str, str, str], tuple[str, str, str]]]]:
    """Sample negative pairs per document.

    Args:
        positives_by_doc: {doc_id: set of (head_span, tail_span)} for positive pairs
        entities_by_doc: {doc_id: [(type, text, span), ...]}
        neg_ratio: number of negatives per positive
        valid_type_pairs: optional set of valid (arg1_type, arg2_type) tuples
        seed: random seed

    Returns:
        {doc_id: [(head_entity, tail_entity), ...]} sampled negative pairs
    """
    rng = random.Random(seed)
    negatives_by_doc: dict[str, list[tuple[tuple[str, str, str], tuple[str, str, str]]]] = {}

    for doc_id, entities in entities_by_doc.items():
        pos_set = positives_by_doc.get(doc_id, set())
        all_pairs = generate_pairs(entities, valid_type_pairs)

        # Filter out positive pairs
        neg_candidates = [
            (e1, e2) for e1, e2 in all_pairs
            if (e1[2], e2[2]) not in pos_set
        ]

        n_positives = len(pos_set)
        n_sample = min(neg_ratio * n_positives, len(neg_candidates))

        if n_sample > 0:
            sampled = rng.sample(neg_candidates, n_sample)
            negatives_by_doc[doc_id] = sampled

    return negatives_by_doc


def convert_split(
    rel_tsv: Path,
    texts_dir: Path,
    output_path: Path,
    lang: str = "english",
    entities_tsv: Path | None = None,
    neg_ratio: int = 0,
    valid_type_pairs: set[tuple[str, str]] | None = None,
    seed: int = 42,
) -> int:
    """Convert one TSV+texts split to OpenNRE JSON lines.

    Returns number of instances written.
    """
    df = pd.read_csv(rel_tsv, sep="\t")
    texts = load_texts(texts_dir)

    # Build positive set and entity inventory for negative sampling
    entities_by_doc = None
    positives_by_doc: dict[str, set[tuple[str, str]]] = {}
    if entities_tsv and neg_ratio > 0:
        entities_by_doc = load_entities(entities_tsv)
        for _, row in df.iterrows():
            doc_id = str(row["document_id"])
            positives_by_doc.setdefault(doc_id, set()).add(
                (str(row["head_span"]), str(row["tail_span"]))
            )

    count = 0
    neg_count = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as f:
        # Write positive instances
        for _, row in df.iterrows():
            doc_id = str(row["document_id"])
            relation = row["relation"]

            if doc_id not in texts:
                skipped += 1
                continue
            if relation not in RELATION_TYPES:
                skipped += 1
                continue

            instance = _make_instance(
                text=texts[doc_id],
                head_type=row["head_type"],
                head_text=row["head_text"],
                head_span=str(row["head_span"]),
                tail_type=row["tail_type"],
                tail_text=row["tail_text"],
                tail_span=str(row["tail_span"]),
                relation=relation,
                doc_id=doc_id,
                lang=lang,
            )
            if instance:
                f.write(json.dumps(instance, ensure_ascii=False) + "\n")
                count += 1

        # Write negative instances
        if entities_by_doc and neg_ratio > 0:
            neg_pairs_by_doc = add_negatives(
                positives_by_doc, entities_by_doc, neg_ratio,
                valid_type_pairs, seed,
            )
            for doc_id, neg_pairs in neg_pairs_by_doc.items():
                if doc_id not in texts:
                    continue
                text = texts[doc_id]
                for head_ent, tail_ent in neg_pairs:
                    h_type, h_text, h_span = head_ent
                    t_type, t_text, t_span = tail_ent
                    instance = _make_instance(
                        text=text,
                        head_type=h_type,
                        head_text=h_text,
                        head_span=h_span,
                        tail_type=t_type,
                        tail_text=t_text,
                        tail_span=t_span,
                        relation="no_relation",
                        doc_id=doc_id,
                        lang=lang,
                    )
                    if instance:
                        f.write(json.dumps(instance, ensure_ascii=False) + "\n")
                        neg_count += 1

    if skipped:
        print(f"  Skipped {skipped} instances (missing text or unknown relation)")
    if neg_count:
        print(f"  Added {neg_count} no_relation negatives (ratio {neg_ratio}:1)")

    return count + neg_count


def convert_blind(
    entity_tsv: Path,
    texts_dir: Path,
    output_path: Path,
    valid_type_pairs: set[tuple[str, str]] | None = None,
    lang: str = "english",
) -> int:
    """Blind mode: generate all candidate pairs from entity TSV.

    Returns number of instances written.
    """
    entities_by_doc = load_entities(entity_tsv)
    texts = load_texts(texts_dir)

    count = 0
    skipped_docs = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for doc_id, entities in sorted(entities_by_doc.items()):
            if doc_id not in texts:
                skipped_docs += 1
                continue

            text = texts[doc_id]
            pairs = generate_pairs(entities, valid_type_pairs)

            for head_ent, tail_ent in pairs:
                h_type, h_text, h_span = head_ent
                t_type, t_text, t_span = tail_ent
                instance = _make_instance(
                    text=text,
                    head_type=h_type,
                    head_text=h_text,
                    head_span=h_span,
                    tail_type=t_type,
                    tail_text=t_text,
                    tail_span=t_span,
                    relation="no_relation",
                    doc_id=doc_id,
                    lang=lang,
                )
                if instance:
                    f.write(json.dumps(instance, ensure_ascii=False) + "\n")
                    count += 1

    if skipped_docs:
        print(f"  Skipped {skipped_docs} documents (no matching text file)")

    return count


def detect_input_type(tsv_path: Path) -> str:
    """Auto-detect TSV type by column names.

    Returns 'relation' or 'entity'.
    """
    df = pd.read_csv(tsv_path, sep="\t", nrows=0)
    columns = set(df.columns)
    if "relation" in columns:
        return "relation"
    if "entity_type" in columns:
        return "entity"
    raise ValueError(
        f"Cannot auto-detect TSV type from columns: {sorted(columns)}. "
        "Expected 'relation' column (relation TSV) or 'entity_type' column (entity TSV)."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert BioNNE-R TSV + text files to OpenNRE JSON lines format."
    )
    parser.add_argument(
        "tsv",
        help="Path to TSV file (auto-detected: relation TSV or entity TSV)",
    )
    parser.add_argument(
        "texts_dir",
        help="Path to directory with .txt files",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output JSON lines file",
    )
    parser.add_argument(
        "--rel2id",
        help="Path to write rel2id.json (default: same directory as output)",
    )
    parser.add_argument(
        "--lang",
        default="english",
        help="NLTK sentence tokenizer language (default: english)",
    )
    parser.add_argument(
        "--entities",
        help="Entity TSV for negative sampling (labeled mode only)",
    )
    parser.add_argument(
        "--neg-ratio",
        type=int,
        default=0,
        help="Negatives per positive for negative sampling (default: 0 = no negatives)",
    )
    parser.add_argument(
        "--config",
        help="Path to annotation_short-bio.conf for type-based pair filtering",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for negative sampling (default: 42)",
    )
    args = parser.parse_args()

    tsv_path = Path(args.tsv)
    texts_dir = Path(args.texts_dir)
    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load config-based type filtering if provided
    valid_type_pairs = None
    if args.config:
        valid_type_pairs = parse_config(args.config)
        print(f"Loaded config: {len(valid_type_pairs)} valid type pairs")

    # Generate rel2id.json (always includes no_relation)
    if args.rel2id:
        rel2id_path = Path(args.rel2id)
    else:
        rel2id_path = output_path.parent / "rel2id.json"

    rel2id = {rel: i for i, rel in enumerate(RELATION_TYPES)}
    rel2id_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rel2id_path, "w") as f:
        json.dump(rel2id, f, indent=2)
    print(f"Wrote {rel2id_path} ({len(rel2id)} classes)")

    # Auto-detect input type
    input_type = detect_input_type(tsv_path)
    print(f"Detected input type: {input_type}")

    if input_type == "entity":
        # Blind mode
        print(f"Blind mode: {tsv_path} + {texts_dir} -> {output_path}")
        n = convert_blind(
            tsv_path, texts_dir, output_path,
            valid_type_pairs=valid_type_pairs,
            lang=args.lang,
        )
        print(f"  {n} candidate pairs")
    else:
        # Labeled mode
        entities_tsv = Path(args.entities) if args.entities else None
        print(f"Converting {tsv_path} + {texts_dir} -> {output_path}")
        n = convert_split(
            tsv_path, texts_dir, output_path,
            lang=args.lang,
            entities_tsv=entities_tsv,
            neg_ratio=args.neg_ratio,
            valid_type_pairs=valid_type_pairs,
            seed=args.seed,
        )
        print(f"  {n} instances")

    print("Done.")


if __name__ == "__main__":
    main()
