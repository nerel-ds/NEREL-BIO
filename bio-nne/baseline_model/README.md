# Baseline Solution


This solution leverages BINDER ([**BI**-encoder for **N**ame**D** **E**ntity **R**ecognition via Contrastive Learning](https://openreview.net/forum?id=9EAQVEINuum)) model[^1].
Based on the bi-encoder representations, BINDER introduces a unified contrastive learning framework for NER, which encourages the representation of entity types to be similar to the corresponding entity mentions, and to be dissimilar with non-entity text spans.


### 1. Data Preparation

Binder framework requires HF-DS JSON files as input for training data. As a first step, we need to convert our NEREL-BIO dataset from BRAT to HFDS format. Here is a [script](brat_to_hfds.py) for it.
```bash
python brat_to_hfds.py --brat_dataset_path data/NEREL-BIO --tags_path data/nerel_bio_frequent.tags --hfds_output_path data/processed_data
```

### 2. Environment Setup
```bash
conda create -n binder -y python=3.9
conda activate binder
conda install pytorch==1.13 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install transformers==4.24.0 datasets==2.6.1 wandb==0.13.5 seqeval==1.2.2
```

### 3. Experiment Run
In this setup we use [Roberta](https://huggingface.co/xlm-roberta-large) as a backbone model[^2].
When you have prepared data for training and validation, converted it from BRAT to HFDS and finished environment setup, you can run the command below to train the model.

```bash
python run_ner.py conf/nerel-bio.json
```

To run experiments with other parameters, simply change the [config](conf/nerel-bio.json).

### 4. Submitting the results

To submit obtained results on Codalab, you need to archive it and upload the .zip file. Please note, that your .zip file should contain only ONE file (your JSON predictions).

You can use the following command to archive the file.

```bash
zip -r submission.zip results -x "*/.*"
```

## References

[^1]: Zhang et al. (2022). 
Optimizing Bi-Encoder for Named Entity Recognition via Contrastive Learning. 
arXiv:2208.14565


[^2]: Conneau, A. et al. (2019). 
Unsupervised Cross-lingual Representation Learning at Scale. 
CoRR, 11(3), 147-148.


