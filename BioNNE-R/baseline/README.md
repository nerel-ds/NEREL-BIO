# BioNNE-R Relation Extraction Baseline

Baseline for the BioNNE-R shared task using [OpenNRE](https://github.com/thunlp/OpenNRE) with `bert-base-multilingual-cased` and entity markers.

## Pretrained Models

Pretrained checkpoints for all three tracks (688 MB each, `bert-base-multilingual-cased`):

| Track | Checkpoint | Dev Macro F1 |
|-------|-----------|--------------|
| English | `eng_model.pth.tar` | 0.6944 |
| Russian | `rus_model.pth.tar` | 0.7166 |
| Bilingual | `bil_model.pth.tar` | — |

Download from [GitHub Releases](https://github.com/nerel-ds/NEREL-BIO/releases/tag/BioNNE-R) and place in `outputs/`.

To predict with a pretrained model (no training needed):

```bash
# Prepare dev data
python prepare_data.py eng-dev-rel.tsv texts/ -o data/eng_dev.txt

# Predict
python baseline.py predict --data data/eng_dev.txt --rel2id data/rel2id.json \
    --ckpt outputs/eng_model.pth.tar -o outputs/eng_pred.tsv

# Evaluate
python score.py --pred outputs/eng_pred.tsv --gold eng-dev-rel.tsv
```

## Quick Start

```bash
# 1. Prepare data (converts relation TSV + raw article texts to OpenNRE JSON-lines format)
#    texts/ = directory of raw .txt article files (one per document)
python prepare_data.py eng-train-rel.tsv texts/ -o data/eng_train.txt --rel2id data/rel2id.json --entities eng-train-ent.tsv --neg-ratio 3

python prepare_data.py eng-dev-rel.tsv texts/ -o data/eng_dev.txt

# 2. Train (model learns 15 classes including no_relation)
python baseline.py train --train data/eng_train.txt --dev data/eng_dev.txt --rel2id data/rel2id.json --ckpt outputs/eng_model.pth.tar

# 3. Predict (generates CodaBench-compatible TSV, no_relation pairs are filtered out)
python baseline.py predict --data data/eng_dev.txt --rel2id data/rel2id.json --ckpt outputs/eng_model.pth.tar -o outputs/eng_pred.tsv

# 4. Evaluate
python score.py --pred outputs/eng_pred.tsv --gold eng-dev-rel.tsv
```

## All Three Tracks

### English

```bash
python prepare_data.py eng-train-rel.tsv texts/ -o data/eng_train.txt \
    --entities eng-train-ent.tsv --neg-ratio 3 --rel2id data/rel2id.json \
    --config annotation_short-bio.conf
python prepare_data.py eng-dev-rel.tsv texts/ -o data/eng_dev.txt
python baseline.py train --train data/eng_train.txt --dev data/eng_dev.txt \
    --rel2id data/rel2id.json --ckpt outputs/eng_model.pth.tar
python baseline.py predict --data data/eng_dev.txt --rel2id data/rel2id.json \
    --ckpt outputs/eng_model.pth.tar -o outputs/eng_pred.tsv
python score.py --pred outputs/eng_pred.tsv --gold eng-dev-rel.tsv
```

### Russian

```bash
python prepare_data.py rus-train-rel.tsv texts/ -o data/rus_train.txt \
    --entities rus-train-ent.tsv --neg-ratio 3 --rel2id data/rel2id.json \
    --config annotation_short-bio.conf
python prepare_data.py rus-dev-rel.tsv texts/ -o data/rus_dev.txt
python baseline.py train --train data/rus_train.txt --dev data/rus_dev.txt \
    --rel2id data/rel2id.json --ckpt outputs/rus_model.pth.tar
python baseline.py predict --data data/rus_dev.txt --rel2id data/rel2id.json \
    --ckpt outputs/rus_model.pth.tar -o outputs/rus_pred.tsv
python score.py --pred outputs/rus_pred.tsv --gold rus-dev-rel.tsv
```

### Bilingual

```bash
# Combine English + Russian data
cat data/eng_train.txt data/rus_train.txt > data/bil_train.txt
cat data/eng_dev.txt data/rus_dev.txt > data/bil_dev.txt
python baseline.py train --train data/bil_train.txt --dev data/bil_dev.txt \
    --rel2id data/rel2id.json --ckpt outputs/bil_model.pth.tar
python baseline.py predict --data data/bil_dev.txt --rel2id data/rel2id.json \
    --ckpt outputs/bil_model.pth.tar -o outputs/bil_pred.tsv
```

## Blind Prediction Workflow

For blind evaluation where participants receive only entity TSV + raw texts (no relation labels):

```bash
# 1. Prepare blind test data from entity TSV (auto-detected)
python prepare_data.py eng-test-ent.tsv texts/ -o data/eng_test.txt \
    --config annotation_short-bio.conf

# 2. Predict (no_relation pairs are filtered from output TSV)
python baseline.py predict --data data/eng_test.txt --rel2id data/rel2id.json \
    --ckpt outputs/eng_model.pth.tar -o outputs/eng_pred.tsv
```

## Files

| File | Description |
|------|-------------|
| `prepare_data.py` | Convert TSV + text files to OpenNRE JSON lines format |
| `baseline.py` | Train and predict using OpenNRE framework |
| `baseline.ipynb` | Interactive notebook version of the full pipeline |
| `patch_opennre.py` | Post-install fix for OpenNRE (encoding, AdamW, num_workers) |
| `score.py` | Evaluation script (macro F1) |

## Usage

### Prepare Data

The input TSV type is auto-detected by column headers:
- Has `relation` column → **labeled mode** (relation TSV)
- Has `entity_type` without `relation` → **blind mode** (entity TSV)

```bash
# Labeled mode (current behavior, no negatives)
python prepare_data.py <rel_tsv> <texts_dir> -o <output.txt> [--rel2id rel2id.json] [--lang english]

# Labeled mode + negative sampling
python prepare_data.py <rel_tsv> <texts_dir> -o <output.txt> --entities <ent.tsv> --neg-ratio 3

# Blind mode (entity TSV → all candidate pairs)
python prepare_data.py <ent_tsv> <texts_dir> -o <output.txt> [--config annotation_short-bio.conf]
```

**Note on file extensions:** Output files use `.txt` extension because that is what OpenNRE expects, but the content is JSON lines (one JSON object per line).

`prepare_data.py` also generates `rel2id.json` — a mapping from relation type strings to integer indices (e.g., `{"ABBREVIATION": 0, ..., "no_relation": 14}`). This file is **not** part of the input data; it is created during data preparation and consumed by `baseline.py` to map between label strings and model output neurons.

`prepare_data.py` extracts the minimal sentence segment around each entity pair (using NLTK punkt), adjusts character offsets, and writes one JSON dict per line:

```json
{"text": "...", "h": {"name": "seizure", "pos": [20, 27]}, "t": {"name": "phenobarbital", "pos": [41, 54]}, "relation": "TREATED_USING", "doc_id": "...", "head_span": "...", ...}
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `tsv` | yes | — | Path to TSV file (relation or entity, auto-detected) |
| `texts_dir` | yes | — | Directory of raw `.txt` article files (one per document) |
| `-o` / `--output` | yes | — | Output JSON lines file |
| `--rel2id` | no | same dir as output | Path to write `rel2id.json` (generated, not input data) |
| `--lang` | no | `english` | NLTK sentence tokenizer language |
| `--entities` | no | — | Entity TSV for negative sampling (labeled mode only) |
| `--neg-ratio` | no | `0` | Negatives per positive (default: 0 = no negatives) |
| `--config` | no | — | `annotation_short-bio.conf` for type-based pair filtering |
| `--seed` | no | `42` | Random seed for negative sampling |

### Train

```bash
python baseline.py train --train <train.txt> --dev <dev.txt> --rel2id <rel2id.json> [--ckpt model.pth.tar] [--epochs 10] ...
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--train` | yes | — | Training data (JSON lines) |
| `--dev` | yes | — | Dev data (JSON lines) |
| `--rel2id` | yes | — | Path to `rel2id.json` |
| `--ckpt` | no | `outputs/model.pth.tar` | Checkpoint save path |
| `--model` | no | `bert-base-multilingual-cased` | Pretrained model name |
| `--max_length` | no | `256` | Maximum sequence length |
| `--batch_size` | no | `16` | Batch size |
| `--lr` | no | `2e-5` | Learning rate |
| `--epochs` | no | `10` | Number of epochs |
| `--warmup_steps` | no | `300` | Learning rate warmup steps |

### Predict

```bash
python baseline.py predict --data <dev.txt> --rel2id <rel2id.json> --ckpt <model.pth.tar> -o <pred.tsv>
```

Predictions where the model outputs `no_relation` are automatically filtered from the output TSV. A summary line prints how many were filtered vs. how many actual relations remain.

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data` | yes | — | Input data (JSON lines) |
| `--rel2id` | yes | — | Path to `rel2id.json` |
| `--ckpt` | no | `outputs/model.pth.tar` | Checkpoint path |
| `-o` / `--output` | no | `outputs/pred.tsv` | Output predictions TSV |
| `--model` | no | `bert-base-multilingual-cased` | Pretrained model name |
| `--max_length` | no | `256` | Maximum sequence length |

### Evaluate

```bash
python score.py --pred outputs/pred.tsv --gold eng-dev-rel.tsv
```

### Notebook

`baseline.ipynb` is a self-contained interactive version of the full pipeline. It includes all data preparation functions inline (no external module imports beyond standard libraries and OpenNRE), so it can run standalone in Jupyter or Colab.

Sections:
1. **Setup** — imports, configuration, `RELATION_TYPES` (15 classes including `no_relation`)
2. **Data Preparation Functions** — all functions from `prepare_data.py` (config parsing, entity loading, pair generation, negative sampling, blind mode)
3. **Data Exploration** — relation distributions, sample instances
4. **Training** — OpenNRE `SentenceRE` framework
5. **Prediction on Dev Set** — predict + filter `no_relation` from output
6. **Evaluation** — per-relation P/R/F1 table
7. **Blind Prediction** — prepare candidate pairs from entity TSV, predict, filter
8. **Error Analysis** — misclassified examples, confusion matrix

Edit the configuration cell (cell 4) to adjust paths, model, neg-ratio, etc.

## Dependencies

Requires **Python 3.10+** (uses PEP 585/604 type hints).

```
opennre
torch>=2.0.0
transformers
nltk
pandas
scikit-learn
```

Install:
```bash
# PyTorch — see https://pytorch.org/get-started/locally/ for your CUDA version
pip install torch

# OpenNRE is NOT on PyPI — install from GitHub
pip install git+https://github.com/thunlp/OpenNRE.git

pip install transformers nltk pandas scikit-learn

# Patch OpenNRE for modern transformers + Windows compatibility
python patch_opennre.py
```

## Output Format

Predictions TSV (`outputs/pred.tsv`) — CodaBench-compatible:

| Column | Description |
|--------|-------------|
| document_id | Document identifier |
| relation | Predicted relation type |
| head_text | Head entity text |
| head_span | Head entity span (start-end) |
| head_type | Head entity type |
| tail_text | Tail entity text |
| tail_span | Tail entity span (start-end) |
| tail_type | Tail entity type |

## Relation Types

ABBREVIATION, ALTERNATIVE_NAME, SUBCLASS_OF, PART_OF, TREATED_USING, ORIGINS_FROM, TO_DETECT_OR_STUDY, AFFECTS, HAS_CAUSE, APPLIED_TO, USED_IN, ASSOCIATED_WITH, PHYSIOLOGY_OF, FINDING_OF

The model also learns a 15th class `no_relation` for rejecting non-relational entity pairs. This class is filtered from prediction output.

## Entity Types

DISO, ANATOMY, CHEM, DEVICE, PHYS, LABPROC, FINDING, INJURY_POISONING
