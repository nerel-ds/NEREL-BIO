#!/usr/bin/env python3
"""
BioNNE-R Relation Extraction Baseline using OpenNRE.

Usage:
    # Train:
    python baseline.py train --train data/train.txt --dev data/dev.txt --rel2id data/rel2id.json

    # Predict:
    python baseline.py predict --data data/dev.txt --rel2id data/rel2id.json --ckpt outputs/model.pth.tar -o outputs/pred.tsv
"""

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path

# Suppress verbose logging from httpx, transformers, OpenNRE
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.utils.loading_report").setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.WARNING)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import opennre
import pandas as pd
import torch


def train(
    train_path: str,
    dev_path: str,
    rel2id_path: str,
    ckpt_path: str = "outputs/model.pth.tar",
    model_name: str = "bert-base-multilingual-cased",
    max_length: int = 256,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    epochs: int = 10,
    warmup_steps: int = 300,
):
    """Train the relation extraction model."""
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    with open(rel2id_path) as f:
        rel2id = json.load(f)

    print(f"Model: {model_name}")
    print(f"Train: {train_path}")
    print(f"Dev: {dev_path}")
    print(f"Classes: {len(rel2id)}")
    print(f"Checkpoint: {ckpt_path}")

    encoder = opennre.encoder.BERTEntityEncoder(
        max_length=max_length,
        pretrain_path=model_name,
    )

    model = opennre.model.SoftmaxNN(
        sentence_encoder=encoder,
        num_class=len(rel2id),
        rel2id=rel2id,
    )

    framework = opennre.framework.SentenceRE(
        model=model,
        train_path=train_path,
        val_path=dev_path,
        test_path=dev_path,
        ckpt=ckpt_path,
        batch_size=batch_size,
        max_epoch=epochs,
        lr=learning_rate,
        opt="adamw",
        warmup_step=warmup_steps,
    )

    framework.train_model(metric="micro_f1")
    print(f"\nBest checkpoint saved to: {ckpt_path}")


def predict(
    data_path: str,
    rel2id_path: str,
    ckpt_path: str,
    output_path: str,
    model_name: str = "bert-base-multilingual-cased",
    max_length: int = 256,
):
    """Generate CodaBench-compatible predictions."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(rel2id_path) as f:
        rel2id = json.load(f)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Data: {data_path}")

    encoder = opennre.encoder.BERTEntityEncoder(
        max_length=max_length,
        pretrain_path=model_name,
    )

    model = opennre.model.SoftmaxNN(
        sentence_encoder=encoder,
        num_class=len(rel2id),
        rel2id=rel2id,
    )

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # Load instances and predict
    instances = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))

    print(f"Predicting {len(instances)} instances...")

    rows = []
    for inst in instances:
        pred_rel, score = model.infer({
            "text": inst["text"],
            "h": {"pos": inst["h"]["pos"]},
            "t": {"pos": inst["t"]["pos"]},
        })

        rows.append({
            "document_id": inst["doc_id"],
            "relation": pred_rel,
            "head_text": inst["h"]["name"],
            "head_span": inst["head_span"],
            "head_type": inst["head_type"],
            "tail_text": inst["t"]["name"],
            "tail_span": inst["tail_span"],
            "tail_type": inst["tail_type"],
        })

    pred_df = pd.DataFrame(rows)

    # Filter out no_relation predictions (pairs the model determined have no relation)
    total_pred = len(pred_df)
    pred_df = pred_df[pred_df["relation"] != "no_relation"]
    filtered = total_pred - len(pred_df)
    print(f"Filtered {filtered} no_relation predictions, {len(pred_df)} relations remaining")

    pred_df.to_csv(output_path, sep="\t", index=False)
    print(f"Predictions saved to: {output_path}")

    # Evaluate against gold labels in data
    id2rel = {v: k for k, v in rel2id.items()}
    all_relations = sorted(rel2id.keys())

    correct = sum(
        1 for inst, row in zip(instances, rows)
        if inst["relation"] == row["relation"]
    )
    total = len(instances)
    print(f"\nAccuracy: {correct}/{total} = {correct/total:.4f}")

    # Per-relation stats
    gold_counts = Counter(inst["relation"] for inst in instances)
    pred_counts = Counter(row["relation"] for row in rows)
    tp_counts = Counter()
    for inst, row in zip(instances, rows):
        if inst["relation"] == row["relation"]:
            tp_counts[inst["relation"]] += 1

    print(f"\n{'Relation':<25} {'P':>8} {'R':>8} {'F1':>8} {'Support':>8}")
    print("-" * 60)
    f1_scores = []
    for rel in all_relations:
        tp = tp_counts.get(rel, 0)
        pred_total = pred_counts.get(rel, 0)
        gold_total = gold_counts.get(rel, 0)
        p = tp / pred_total if pred_total else 0
        r = tp / gold_total if gold_total else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        if gold_total > 0:
            f1_scores.append(f1)
        print(f"{rel:<25} {p:>8.4f} {r:>8.4f} {f1:>8.4f} {gold_total:>8}")

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    print(f"\nMacro F1: {macro_f1:.4f}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="BioNNE-R Relation Extraction Baseline (OpenNRE)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train subcommand ---
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--train", required=True, dest="train_path",
        help="Path to training data (JSON lines)",
    )
    train_parser.add_argument(
        "--dev", required=True, dest="dev_path",
        help="Path to dev data (JSON lines)",
    )
    train_parser.add_argument(
        "--rel2id", required=True,
        help="Path to rel2id.json",
    )
    train_parser.add_argument(
        "--ckpt", default="outputs/model.pth.tar",
        help="Path to save best checkpoint (default: outputs/model.pth.tar)",
    )
    train_parser.add_argument(
        "--model", default="bert-base-multilingual-cased",
        help="Pretrained model (default: bert-base-multilingual-cased)",
    )
    train_parser.add_argument(
        "--max_length", type=int, default=256,
        help="Max sequence length (default: 256)",
    )
    train_parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size (default: 16)",
    )
    train_parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of epochs (default: 10)",
    )
    train_parser.add_argument(
        "--warmup_steps", type=int, default=300,
        help="Warmup steps (default: 300)",
    )

    # --- predict subcommand ---
    pred_parser = subparsers.add_parser("predict", help="Generate predictions")
    pred_parser.add_argument(
        "--data", required=True, dest="data_path",
        help="Path to input data (JSON lines)",
    )
    pred_parser.add_argument(
        "--rel2id", required=True,
        help="Path to rel2id.json",
    )
    pred_parser.add_argument(
        "--ckpt", default="outputs/model.pth.tar",
        help="Path to checkpoint (default: outputs/model.pth.tar)",
    )
    pred_parser.add_argument(
        "-o", "--output", default="outputs/pred.tsv",
        help="Output predictions TSV (default: outputs/pred.tsv)",
    )
    pred_parser.add_argument(
        "--model", default="bert-base-multilingual-cased",
        help="Pretrained model (default: bert-base-multilingual-cased)",
    )
    pred_parser.add_argument(
        "--max_length", type=int, default=256,
        help="Max sequence length (default: 256)",
    )

    args = parser.parse_args()

    if args.command == "train":
        train(
            train_path=args.train_path,
            dev_path=args.dev_path,
            rel2id_path=args.rel2id,
            ckpt_path=args.ckpt,
            model_name=args.model,
            max_length=args.max_length,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            epochs=args.epochs,
            warmup_steps=args.warmup_steps,
        )
    elif args.command == "predict":
        predict(
            data_path=args.data_path,
            rel2id_path=args.rel2id,
            ckpt_path=args.ckpt,
            output_path=args.output,
            model_name=args.model,
            max_length=args.max_length,
        )

    return 0


if __name__ == "__main__":
    exit(main())
