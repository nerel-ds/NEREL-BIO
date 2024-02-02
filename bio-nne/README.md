# [BioNNE task on Nested Named Entity Recognition](http://participants-area.bioasq.org/general_information/BioNNE/)

## Introduction

This is the repository for BioNNE (Biomedical Nested Named Entity Recognition) task. This task is a part of ([BioASQ](http://bioasq.org/)) Workshop that will be held at [CLEF 2024](https://clef2024.imag.fr/).

BioNNE involves NLP challenges on biomedical nested named entity recognition (NER) systems for English and Russian languages.

The evaluation framework is divided into three broad tracks:

*    Track 1 - Bilingual: Participants in this track are required to train a single multi-lingual NER model using training data for both Russian and English languages. The model should be used to generate prediction files for each language's dataset. 

*    Track 2 - English-oriented: Participants in this track are required to train a nested NER model for English scientific abstracts in the biomedical domain. Participants are allowed to train any model architecture on any publicly available data in order to achieve the best performance. 

*    Track 3 - Russian-oriented: Participants in this track are required to train a nested NER model for Russian scientific abstracts in the biomedical domain. Participants are allowed to train any model architecture on any publicly available data in order to achieve the best performance.

<img src="annotation_example.png" width="450">
Nested Named Entity Annotation example

## Dataset
In order to download our train/dev splits with annotations for BioNNE 2024, please register at [BioASQ](http://participants-area.bioasq.org/#) website.

## Baseline Solution
You can find the baseline solution in the [baseline_model](https://github.com/nerel-ds/NEREL-BIO/tree/master/bio-nne/baseline_model) directory.

## Submitting the results
You need to register on the official [BioNNE Codalab page](https://codalab.lisn.upsaclay.fr/competitions/16464) in order to submit your results and see the leaderboard. Please note that your submission should only include the JSON with your predictions archived in .zip format.

## Timeline
Phase |	Dates
--- | --
Training Data Release |	January 2024
Validation Data Release |	TBA
Validation set submission due | TBA
Test data release, evaluation phase starts | TBA
Test set predictions due | TBA
Test set evaluation scores release | TBA
System descriptions due | TBA
Acceptance notification | TBA
Camera-ready system descriptions | TBA
BioASQ Workshop at CLEF 2024 | September 9-12, 2024


## Organizers
Vera Davydova, Sber AI, email: veranchos@gmail.com

Dr. Elena Tutubalina, Artificial Intelligence Research Institute (AIRI), email: tutubalinaev@gmail.com

Dr. Natalia Loukachevitch, Research Computing Center of Lomonosov Moscow State University (MSU), email: louk_nat@mail.ru
