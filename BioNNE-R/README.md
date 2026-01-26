# BioNNE-R Shared Task at BioASQ 2026

The repository is devoted to the **BioNNE-R** (**Bio**medical **N**ested **N**amed **E**ntity for Nested **R**elation Extraction in Russian and English) Shared Task within the [BioASQ 2026 Workshop](https://www.bioasq.org/workshop2026) co-located with [CLEF 2026](https://clef2026.clef-initiative.eu/).

![Alt Text](bionne-r-demo.png)

## Shared Task Overview

The **BioNNE-R Shared Task** addresses the NLP challenge of relation extraction involving nested named entities, i.e. entities that contain other entities within their boundaries. Participants in this task must develop models for nested relation extraction, which can be either language-oriented or bilingual, depending on the track of the task. 

**Goal:** extract relations between annotated nested biomedical entity mentions.

**Data:** Entities from English and Russian scientific abstracts in the biomedical domain and relations between them. The BioNNE-R task utilizes the MCN annotation of the NEREL-BIO dataset [1], which provides annotated mentions of disorders, anatomical structures, chemicals, diagnostic procedures, and biological functions. We design our data to account for a complex structure of nested entity mentions and the partial nature of medical terminology. 

Participants are allowed to train any model architecture on any publicly available data to achieve the best performance.

## **Evaluation Tracks** 

Similar to the [BioNNE 2025 task](https://ceur-ws.org/Vol-4038/paper_3.pdf) [2], the evaluation is structured into **Three Subtasks** under **Two Evaluation Tracks**:

* Two **Monolingual Tracks** requiring separate models for English (**Subtask 1**) and Russian (**Subtask 2**);
    
* **Bilingual Track**: requiring a single model trained on multilingual dataset combined from English and Russian data (**Subtask 3**). Please note that predictions from any mono-lingual model are not allowed in this track.

## **Shared Task-Specific Challenges:**

* **Nestedness of named entities**: usually relation extraction is involving only one level of named entity mentions; in this task, relations are defined for nested ones, i.e. relations could not only span the high-level entities, but also introduce more complex connections like inner entity - outer entity. 

## Participation

To participate in BioNNE-R, please register at the BioASQ website: [https://participants-area.bioasq.org/general_information/general_information_registration/](https://participants-area.bioasq.org/general_information/general_information_registration/). You can join anytime from Feb, 2026 onwards. 

Competition can be found **here**.

## Data

### Data Overview

| | TSV-formatted entity data | TSV-formatted relation data | Raw Texts |
| --- | --- | --- | --- |
| English | [train](https://github.com/nerel-ds/NEREL-BIO/blob/master/BioNNE-R/data/en/train/eng-train-ent.tsv), dev **TBD**, test **TBD** | [train](https://github.com/nerel-ds/NEREL-BIO/blob/master/BioNNE-R/data/en/train/eng-train-rel.tsv), , dev **TBD**, test **TBD** | [train](https://github.com/nerel-ds/NEREL-BIO/tree/master/BioNNE-R/data/en/train/texts), dev **TBD**, test **TBD** |
| Russian | [train](https://github.com/nerel-ds/NEREL-BIO/blob/master/BioNNE-R/data/ru/train/rus-train-ent.tsv), dev **TBD**, test **TBD** | [train](https://github.com/nerel-ds/NEREL-BIO/blob/master/BioNNE-R/data/ru/train/rus-train-rel.tsv), dev **TBD**, test **TBD** | [train](https://github.com/nerel-ds/NEREL-BIO/tree/master/BioNNE-R/data/ru/train/texts), dev **TBD**, test **TBD** | 


### Annotated Data Format

Each line of TSV-formatted relation data describes a single biomedical relation of two given entities. For this competition, we will only use several original entity types from NEREL-BIO dataset, namely **ANATOMY**, **CHEM**, **DEVICE** (merged with original **PRODUCT**), **DISO**, **FINDING**, **INJURY_POISONING**, **LABPROC** and **PHYS**. Possible relations classes are also utilized from the NEREL-BIO dataset: **ABBREVIATION**, **AFFECTS**, **ALTERNATIVE_NAME**, **APPLIED_TO**, **ASSOCIATED_WITH**, **FINDING_OF**, **HAS_CAUSE**, **ORIGINS_FROM**, **PART_OF**, **PHYSIOLOGY_OF**, **SUBCLASS_OF**, **TO_DETECT_OR_STUDY**, **TREATED_USING**, **USED_IN**.

#### Entity Data

TSV-format of entity data is `document_id`, `entity_type`, `entity_text` and `entity_span`. 

* `document_id` is a unique textual document identifier the given entity is derived from. Each document contains multiple entities described with their `entity_span` in the document;

* `entity_text` is a textual mention string of the given entity;

* `entity_type` can take one of these values: **ANATOMY**, **CHEM**, **DEVICE** (merged with original **PRODUCT**), **DISO**, **FINDING**, **INJURY_POISONING**, **LABPROC** and **PHYS**. 

* `entity_span` provides a list of comma-separated entity positions within the given textual document. Each span entry provides starting and ending positions, e.g., `22-28`. An entity provided with multiple positions (e.g., `472-476,492-500` for lung injuries) corresponds to an interrupted entity with non-entity words inserted between entity words;

Here are some entity data examples:

```
document_id   entity_type   entity_text             entity_span
---------------------------------------------------------------
25591652_en   CHEM          CBZ                     645-648
26485778_en   DISO          atopic dermatitis       416-433
26978050_ru   FINDING       повышенный уровень АП   1838-1859
27100547_ru   ANATOMY       анастомоза              901-911
```

#### Relation Data

TSV-format of relation (and, consequitively, predictions) is as follows: `document_id`, `relation`, `head_text`, `head_span`, `head_type`, `tail_text`, `tail_span`, `tail_type`.

* `document_id`: is a unique textual document identifier the given relation and entities are derived from. This always matches the `document_id` given in entity data.

* `relation`: relation type between two given entities. One of **ABBREVIATION**, **AFFECTS**, **ALTERNATIVE_NAME**, **APPLIED_TO**, **ASSOCIATED_WITH**, **FINDING_OF**, **HAS_CAUSE**, **ORIGINS_FROM**, **PART_OF**, **PHYSIOLOGY_OF**, **SUBCLASS_OF**, **TO_DETECT_OR_STUDY**, **TREATED_USING**, **USED_IN**. *In your submissions, this field should be predicted by your solution.*

* `head_text`: a textual mention string of the given subject entity, 'head' of the relation. This text always matches `entity_text` in the same document given in entity data.

* `head_span`: provides a list of comma-separated subject entity ('head' of the relation) positions within the given textual document. Each span entry provides starting and ending positions, e.g., `22-28`. An entity provided with multiple positions (e.g., `472-476,492-500` for lung injuries) corresponds to an interrupted entity with non-entity words inserted between entity words. This always matches the `entity_span` in the entity data.

* `head_type`: can take one of these values: **ANATOMY**, **CHEM**, **DEVICE** (merged with original **PRODUCT**), **DISO**, **FINDING**, **INJURY_POISONING**, **LABPROC** and **PHYS**, corresponding to subject entity ('head' of the relation). Matches with `entity_type` given in entity data.

* `tail_text`: a textual mention string of the given object entity, 'tail' of the relation. This text always matches `entity_text` in the same document given in entity data. 

* `tail_span`: provides a list of comma-separated object entity ('tail' of the relation) positions within the given textual document. Each span entry provides starting and ending positions, e.g., `22-28`. An entity provided with multiple positions (e.g., `472-476,492-500` for lung injuries) corresponds to an interrupted entity with non-entity words inserted between entity words. This always matches the `entity_span` in the entity data.

* `tail_type`: can take one of these values: **ANATOMY**, **CHEM**, **DEVICE** (merged with original **PRODUCT**), **DISO**, **FINDING**, **INJURY_POISONING**, **LABPROC** and **PHYS**, corresponding to object entity ('tail' of the relation). Matches with `entity_type` given in entity data.

Here are some relation data examples:

```
document_id   relation           head_text             head_span   head_type   tail_text      tail_span   tail_type
-------------------------------------------------------------------------------------------------------------------
25591652_en   HAS_CAUSE          aggravated absences   649-668     FINDING     CBZ            645-648     CHEM
26485778_en   SUBCLASS_OF        atopic dermatitis     416-433     DISO        dermatitis     423-433     DISO
26977721_ru   ALTERNATIVE_NAME   кофеиновая            134-144     CHEM        кофеином       359-367     CHEM
26356618_ru   PHYSIOLOGY_OF      содержание            319-329     PHYS        плазме крови   295-307     ANATOMY
```

## Baseline Solution

TBD

## Evaluation

### Evaluation Restrictions

1. For Track 2 (Multilingual), predictions from any mono-lingual model are not allowed.
2. For Track 1 (Russian/English), participants are required to treat each language as a separate task. **Distinct models and prediction files are necessary for English and Russian.**
3. Prediction files between two tracks should not match.

### Submission Format

Submission format perfectly aligns with relation data format. Fields `document_id`, `head_text`, `head_span`, `head_type`, `tail_text`, `tail_span`, `tail_type` should always match the ones given in entity data. They will be used to match the ones in the labeled true data for evaluation. `relation` field should contain the prediction for possible entity pairs - one of the **ABBREVIATION**, **AFFECTS**, **ALTERNATIVE_NAME**, **APPLIED_TO**, **ASSOCIATED_WITH**, **FINDING_OF**, **HAS_CAUSE**, **ORIGINS_FROM**, **PART_OF**, **PHYSIOLOGY_OF**, **SUBCLASS_OF**, **TO_DETECT_OR_STUDY**, **TREATED_USING**, **USED_IN**. If no relation is predicted between two entity pairs, it should not appeat in the submission tsv file.

### Evaluation metrics

We treat the shared task as multi-classification task. We choose F1-score for evaluation of submissions, macro-averaged over all relation types. 

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