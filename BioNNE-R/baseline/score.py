#!/usr/bin/env python3
"""
Evaluation script for BioNNE-R relation extraction.

Computes:
- Per-relation Precision, Recall, F1
- Macro F1 (primary metric)
- Micro F1
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_predictions(pred_path: Path) -> pd.DataFrame:
    """Load predictions TSV file.

    Expected columns (same 8-column CodaBench format):
    - document_id, relation, head_text, head_span, head_type,
      tail_text, tail_span, tail_type
    """
    return pd.read_csv(pred_path, sep="\t")


def load_gold(gold_path: Path) -> pd.DataFrame:
    """Load gold TSV file.

    Expected columns:
    - document_id, relation, head_text, head_span, head_type,
      tail_text, tail_span, tail_type
    """
    return pd.read_csv(gold_path, sep="\t")


def create_instance_key(doc_id: str, head_span: str, tail_span: str) -> str:
    """Create unique key for a relation instance."""
    return f"{doc_id}|{head_span}|{tail_span}"


def evaluate(pred_df: pd.DataFrame, gold_df: pd.DataFrame) -> dict:
    """Evaluate predictions against gold standard.

    Returns dict with:
    - per_relation: {relation: {precision, recall, f1, support}}
    - macro_f1: Macro-averaged F1
    - micro_f1: Micro-averaged F1
    - accuracy: Overall accuracy
    """
    # Build gold lookup
    gold_relations = {}
    relation_counts = defaultdict(int)

    for _, row in gold_df.iterrows():
        key = create_instance_key(
            str(row["document_id"]),
            str(row["head_span"]),
            str(row["tail_span"])
        )
        gold_relations[key] = row["relation"]
        relation_counts[row["relation"]] += 1

    # Get all relation types
    all_relations = sorted(set(gold_df["relation"].unique()))

    # Count TP, FP, FN per relation
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    pred_relations = {}
    for _, row in pred_df.iterrows():
        key = create_instance_key(
            str(row["document_id"]),
            str(row["head_span"]),
            str(row["tail_span"])
        )
        pred_relations[key] = row["relation"]

    # Calculate TP and FP
    for key, pred_rel in pred_relations.items():
        if key in gold_relations:
            gold_rel = gold_relations[key]
            if pred_rel == gold_rel:
                tp[pred_rel] += 1
            else:
                fp[pred_rel] += 1
                fn[gold_rel] += 1
        else:
            fp[pred_rel] += 1

    # Calculate FN for gold instances not in predictions
    for key, gold_rel in gold_relations.items():
        if key not in pred_relations:
            fn[gold_rel] += 1

    # Calculate metrics per relation
    per_relation = {}
    for rel in all_relations:
        precision = tp[rel] / (tp[rel] + fp[rel]) if (tp[rel] + fp[rel]) > 0 else 0
        recall = tp[rel] / (tp[rel] + fn[rel]) if (tp[rel] + fn[rel]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_relation[rel] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": relation_counts[rel],
            "tp": tp[rel],
            "fp": fp[rel],
            "fn": fn[rel],
        }

    # Calculate macro F1
    f1_scores = [per_relation[rel]["f1"] for rel in all_relations]
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    # Calculate micro F1
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0 else 0
    )

    # Calculate accuracy
    correct = sum(
        1 for key, pred_rel in pred_relations.items()
        if key in gold_relations and pred_rel == gold_relations[key]
    )
    total = len(gold_relations)
    accuracy = correct / total if total > 0 else 0

    return {
        "per_relation": per_relation,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "accuracy": accuracy,
        "total_instances": total,
    }


def print_classification_report(results: dict):
    """Print classification report in scikit-learn style."""
    print("\nClassification Report:")
    print("=" * 80)
    print(f"{'Relation':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 80)

    per_relation = results["per_relation"]
    for rel in sorted(per_relation.keys()):
        metrics = per_relation[rel]
        print(
            f"{rel:<25} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f} "
            f"{metrics['support']:>10}"
        )

    print("-" * 80)
    print(f"{'Macro avg':<25} {'-':>10} {'-':>10} {results['macro_f1']:>10.4f} {results['total_instances']:>10}")
    print(f"{'Micro avg':<25} {results['micro_precision']:>10.4f} {results['micro_recall']:>10.4f} {results['micro_f1']:>10.4f} {results['total_instances']:>10}")
    print("=" * 80)
    print(f"\nMacro F1 (primary metric): {results['macro_f1']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BioNNE-R relation extraction predictions."
    )
    parser.add_argument(
        "--pred",
        required=True,
        help="Path to predictions TSV file"
    )
    parser.add_argument(
        "--gold",
        required=True,
        help="Path to gold standard TSV file"
    )
    parser.add_argument(
        "--output",
        help="Path to save results JSON (optional)"
    )

    args = parser.parse_args()

    pred_path = Path(args.pred)
    gold_path = Path(args.gold)

    if not pred_path.exists():
        print(f"Error: Predictions file not found: {pred_path}")
        return 1

    if not gold_path.exists():
        print(f"Error: Gold file not found: {gold_path}")
        return 1

    print(f"Predictions: {pred_path}")
    print(f"Gold: {gold_path}")

    pred_df = load_predictions(pred_path)
    gold_df = load_gold(gold_path)

    print(f"\nPredictions: {len(pred_df)} instances")
    print(f"Gold: {len(gold_df)} instances")

    results = evaluate(pred_df, gold_df)
    print_classification_report(results)

    if args.output:
        import json
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
