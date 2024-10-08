# Concept Normalization over Nested Entities

This folder is devoted to a novel dataset for nested entity linking in Russian that is described in "Biomedical Concept Normalization over Nested Entities with Partial UMLS Terminology in Russian" paper accepted to LREC-COLING 2024. Here, we release our data as well as source code for preprocessing and simple baselines.

Here is our poster presented at COLING 2024 (see ![paper](https://aclanthology.org/2024.lrec-main.213.pdf)):
<p align="center">
<img src="https://github.com/nerel-ds/NEREL-BIO/blob/master/nested-mcn/COLING_NEREL_BIO_POSTER.png" width="800">
</p>


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
For evaluation of a fine-tuned BioSyn model on the test, you can run:

```bash
mkdir eval_results_biosyn/sapbert/checkpoint_20/
python biosyn/eval.py \
    --model_name_or_path pretrained_biosyn/sapbert/checkpoint_20/ \
    --dictionary_path nested-mcn/data/dictionary/vocab_umls_rus_biosyn.txt \
    --data_dir nested-mcn/data/biosyn_format/random_split/test/ \
    --output_dir eval_results_biosyn/sapbert/checkpoint_20/ \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions \
    --score_mode hybrid
```

### Nested normalization baselines

Originally, BioSyn adopts two similarity scores to iteratively update candidates given a mention:

(i) a dot-product between BERT embeddings of a mention and a candidate;

(ii) a dot-product between TF-IDF embeddings of a mention and a candidate.

In our paper, we propose two simple baselines for nested entity normalization that build upon Biosyn:

1. Reranking baseline: a small fully-connected network is trained to rerank top candidates produced by frozen BioSyn;

2. Addition of the third BERT-based similarity score to Biosyn that takes a concatenation of a mention $m$ and the longest mention $m'$ such that $m \in m'$.

Please see [training](https://github.com/nerel-ds/NEREL-BIO/tree/master/nested-mcn/sh/train) and [evaluation](https://github.com/nerel-ds/NEREL-BIO/tree/master/nested-mcn/sh/eval) scripts for these baselines.

### Citation
Loukachevitch, N., Sakhovskiy, A., & Tutubalina, E. (2024, May). Biomedical Concept Normalization over Nested Entities with Partial UMLS Terminology in Russian. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024) (pp. 2383-2389).
https://aclanthology.org/2024.lrec-main.213.pdf

```
@CONFERENCE{Loukachevitch20242383,
	author = {Loukachevitch, Natalia and Sakhovskiy, Andrey and Tutubalina, Elena},
	title = {Biomedical Concept Normalization over Nested Entities with Partial UMLS Terminology in Russian},
	year = {2024},
	journal = {2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation, LREC-COLING 2024 - Main Conference Proceedings},
	pages = {2383 – 2389},
}
```
See also
```
@article{NERELBIO,
    author = {Loukachevitch, Natalia and Manandhar, Suresh and Baral, Elina and Rozhkov, Igor and Braslavski, Pavel and Ivanov, Vladimir and Batura, Tatiana and Tutubalina, Elena},
    title = "{NEREL-BIO: A Dataset of Biomedical Abstracts Annotated with Nested Named Entities}",
    journal = {Bioinformatics},
    year = {2023},
    month = {04},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btad161},
    url = {https://doi.org/10.1093/bioinformatics/btad161},
    note = {btad161},
}
```
