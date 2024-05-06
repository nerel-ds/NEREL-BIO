# Concept Normalization over Nested Entities

This folder is devoted to a novel dataset for nested entity linking in Russian that is described in "Biomedical Concept Normalization over Nested Entities with Partial UMLS Terminology in Russian" paper accepted to LREC-COLING 2024. Here, we release our data as well as source code for preprocessing and simple baselines.



## Data

We release biomedical entity normalization annotation over Russian nested entities in two formats:

* [BRAT format](https://github.com/nerel-ds/NEREL-BIO/tree/master/nested-mcn/data/brat);

* [BioSyn-compatible format](https://github.com/nerel-ds/NEREL-BIO/tree/master/nested-mcn/data/biosyn_format). For more details, see [BioSyn repository](https://github.com/dmis-lab/BioSyn)


Entities are normalized to [UMLS metathesaurus](https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/index.html). In UMLS, a  biomedical concept is assigned a Concept Unique Identifier (CUI) and a list of concept names in multiple languages.
Normalization dictionary (derived from the Russian UMLS subpart) is available [here](https://github.com/nerel-ds/NEREL-BIO/blob/master/nested-mcn/data/dictionary/vocab_umls_rus_biosyn.txt). Each line of the dictionary of <CUI, concept name> pairs separated with "||" string.

## Evaluation

### Zero-shot evaluation

To run non-nested evaluation on our dataset, you can run a evaluation script from [Fair-Evaluation-BERT repository](https://github.com/alexeyev/Fair-Evaluation-BERT.git):

```bash
cd Fair-Evaluation-BERT/
eval_bert_ranking.py \
--model_dir ${MODEL_NAME} \
--data_folder ../data/biosyn_format/random_split/test/ \
--vocab ../data/dictionary/vocab_umls_rus_biosyn.txt

```

### Supervised evaluation

To train on our corpus in a non-nested approach (i.e., considering each entity regardless of whether it is nested within another entity or has another entity as its subpart), you can adopt [BioSyn](https://github.com/dmis-lab/BioSyn).

For instance, you can run finetuning as follows:
```bash
mkdir pretrained_biosyn/sapbert
python biosyn/train.py \
--model_name_or_path cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR \
--train_dictionary_path nested-mcn/data/dictionary/vocab_umls_rus_biosyn.txt \
--train_dir nested-mcn/data/biosyn_format/random_split/train/ \
--output_dir pretrained_biosyn/sapbert \
--use_cuda \
--topk 20 \
--epoch 20 \
--train_batch_size 16 \
--initial_sparse_weight 0 \
--learning_rate 1e-5 \
--max_length 25 \
--dense_ratio 0.5

```

Originally, BioSyn adopts two similarity scores to iteratively update candidates given a mention:

(i) a dot-product between BERT embeddings of a mention and a candidate;

(ii) a dot-product between TF-IDF embeddings of a mention and a candidate.

In our paper, we propose two simple baselines for nested entity normalization that build upon Biosyn:

1. Reranking baseline: a small fully-connected network is trained to rerank top candidates produced by frozen BioSyn;

2. Addition of the third BERT-based similarity score to Biosyn that takes a concatenation of a mention $m$ and the longest mention $m'$ such that $m \in m'$.




