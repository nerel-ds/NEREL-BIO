# Concept Normalization over Nested Entities

This folder is devoted to a novel dataset for nested entity linking in Russian that is described in "Biomedical Concept Normalization over Nested Entities with Partial UMLS Terminology in Russian" paper accepted to LREC-COLING 2024. Here, we release our data as well as source code for preprocessing and simple baselines.



## Data

We release biomedical entity normalization annotation over Russian nested entities in two formats:

* [BRAT format](https://github.com/nerel-ds/NEREL-BIO/tree/master/nested-mcn/data/brat);

* [BioSyn](https://github.com/dmis-lab/BioSyn)-compatible format.

Entities are normalized to [UMLS metathesaurus](https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/index.html). In UMLS, a  biomedical concept is assigned a Concept Unique Identifier (CUI) and a list of concept names in multiple languages.
Normalization dictionary (derived from the Russian UMLS subpart) is available [here](https://github.com/nerel-ds/NEREL-BIO/blob/master/nested-mcn/data/dictionary/vocab_umls_rus_biosyn.txt). Each line of the dictionary of <CUI, concept name> pairs separated with "||" string.

## Nested normalization baselines

Source code for our nested entity normalization is available at [nelbio/](https://github.com/nerel-ds/NEREL-BIO/tree/master/nested-mcn/nelbio) directory. 


## Evaluation

To run non-nested evaluation on our dataset, you can run a evaluation script from [Fair-Evaluation-BERT repository](https://github.com/alexeyev/Fair-Evaluation-BERT.git):

```bash
cd Fair-Evaluation-BERT/
eval_bert_ranking.py --model_dir ${MODEL_NAME} \
                         --data_folder ../data/biosyn_format/random_split/test/ \
                         --vocab ../data/dictionary/vocab_umls_rus_biosyn.txt

```



