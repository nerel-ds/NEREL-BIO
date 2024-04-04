## Concept Normalization over Nested Entities

We release biomedical entity normalization annotation over Russian nested entities in two formats:

* [BRAT format](https://github.com/nerel-ds/NEREL-BIO/tree/master/nested-mcn/data/brat);

* [BioSyn format](https://github.com/dmis-lab/BioSyn). 

Normalization dictionary is available [here](https://github.com/nerel-ds/NEREL-BIO/blob/master/nested-mcn/data/dictionary/vocab_umls_rus_biosyn.txt).

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



