# BioNNE-R Shared Task at BioASQ 2026

The repository is devoted to the **BioNNE-R** (**Bio**medical **N**ested **N**amed **E**ntity for Nested **R**elation Extraction in Russian and English) Shared Task within the [BioASQ 2026 Workshop](https://www.bioasq.org/workshop2026) co-located with [CLEF 2026](https://clef2026.clef-initiative.eu/).

![Alt Text](bionne-r-demo.png)

## Shared Task Overview

The **BioNNE-R Shared Task** addresses the NLP challenge of relation extraction involving nested named entities, i.e. entities that contain other entities within their boundaries. Participants in this task must develop models for nested relation extraction, which can be either language-oriented or bilingual, depending on the track of the task. 

**Goal:** extract relations between annotated nested biomedical entity mentions.

**Data:** Entities from English and Russian scientific abstracts in the biomedical domain and relations between them. The BioNNE-R task utilizes the MCN annotation of the NEREL-BIO dataset [1], which provides annotated mentions of disorders, anatomical structures, chemicals, diagnostic procedures, and biological functions. We design our data to account for a complex structure of nested entity mentions and the partial nature of medical terminology. 

Participants are allowed to train any model architecture on any publicly available data to achieve the best performance.

**Evaluation Tracks:** Similar to the [BioNNE 2025 task](https://ceur-ws.org/Vol-4038/paper_3.pdf) [2], the evaluation is structured into **Three Subtasks** under **Two Evaluation Tracks**:

* Two **Monolingual Tracks** requiring separate models for English (**Subtask 1**) and Russian (**Subtask 2**);
    
* **Bilingual Track**: requiring a single model trained on multilingual dataset combined from English and Russian data (**Subtask 3**). Please note that predictions from any mono-lingual model are not allowed in this track.

**Shared Task-Specific Challenges:**

* **Nestedness of named entities**: usually relation extraction is involving only one level of named entity mentions; in this task, relations are defined for nested ones, i.e. relations could not only span the high-level entities, but also introduce more complex connections like inner entity - outer entity. 

## Participation

You can join anytime from Feb, 2026 onwards.

## Data

TBD

### Annotated Data Format

TBD 

## Baseline Solution

TBD

## Evaluation

### Evaluation Restrictions

1. For Track 2 (Multilingual), predictions from any mono-lingual model are not allowed.
2. For Track 1 (Russian/English), participants are required to treat each language as a separate task. **Distinct models and prediction files are necessary for English and Russian.**
3. Prediction files between two tracks should not match.

### Submission Format

TBD

### Evaluation metrics

TBD

## Important Dates:


| Phase                                      | Date             |
|--------------------------------------------|------------------|
| Training Data Release                       | Feb 2026       |
| Dev data release, Development phase start   | TBD      |
| Test data release, Evaluation phase start | TBD    |
| Test set predictions due                    | TBD       |
| Submission of participant papers            | TBD      |
| Acceptance notification for participant papers | TBD     |
| Camera-ready working notes papers         | TBD      |
| **BioASQ Workshop** at [CLEF 2026](https://clef2026.clef-initiative.eu)              | September 21-24, 2026 |


## References

[1] Loukachevitch, Natalia, Andrey Sakhovskiy, and Elena Tutubalina. [Biomedical Concept Normalization over Nested Entities with Partial UMLS Terminology in Russian](https://aclanthology.org/2024.lrec-main.213). Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024). 2024.

[2] Sakhovskiy, Andrey, Natalia Loukachevitch, and Elena Tutubalina. [Overview of the BioASQ BioNNE-L Task on Biomedical
Nested Entity Linking in CLEF 2025](https://ceur-ws.org/Vol-4038/paper_3.pdf). CLEF Working Notes (2025).