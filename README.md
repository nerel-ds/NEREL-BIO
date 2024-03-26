# NEREL-BIO: Biomedical Corpus for Nested Named Entity Recognition

This project presents NEREL-BIO -- an annotation scheme and corpus of PubMed abstracts in Russian and in English. NEREL-BIO extends the general domain dataset [NEREL](https://github.com/nerel-ds/NEREL). [NEREL-BIO annotation scheme](https://github.com/nerel-ds/NEREL-BIO/blob/master/nerel-bio-guidelines.pdf) covers both general and biomedical domains making it suitable for domain transfer experiments. 

<img src="nerel-bio.png" width="450">




### List of entity types

|No. | Entity type | No. | Entity type | No. | Entity type
|---|---|---|---|---|---
|1. | ACTIVITY | 14. | MEDPROC | 27. | MONEY
|2. | ADMINISTRATION_ROUTE | 15. | MENTALPROC | 28. | NATIONALITY
|3. | ANATOMY | 16. | PHYS | 29. | NUMBER
|4. | CHEM | 17. | SCIPROC | 30. | ORDINAL
|5. | DEVICE | 18. | AGE | 31. | ORGANIZATION
|6. | DISO | 19. | CITY | 32. | PERCENT
|7. | FINDING | 20. | COUNTRY | 33. | PERSON
|8. | FOOD | 21. | DATE | 34. | PRODUCT
|9. | GENE | 22. | DISTRICT | 35. | PROFESSION
|10. | INJURY_POISONING | 23. | EVENT | 36. | STATE_OR_PROVINCE
|11. | HEALTH_CARE_ACTIVITY | 24. | FAMILY | 37. | TIME
|12. | LABPROC | 25. | FACILITY |  | 
|13. | LIVB | 26. | LOCATION |  | 

[![](https://img.shields.io/badge/AREkit--ss_Compatible-0.23.1-purple.svg)](https://github.com/nicolay-r/arekit-ss#usage)

> 📓 **Update 1 November 2023**: this collection **is now available in [arekit-ss](https://github.com/nicolay-r/arekit-ss)**
> for a [quick sampling](https://github.com/nicolay-r/arekit-ss#usage) of contexts with most subject-object relation mentions with just **single script into
> `JSONL/CSV/SqLite`** including (optional) language transfering 🔥 [[Learn more ...]](https://github.com/nicolay-r/arekit-ss#usage)

### Concept Normalization over Nested Entities

We release entity normalization annotation over nested entities, [see](https://github.com/nerel-ds/NEREL-BIO/tree/master/nested-mcn/data).

### Baselines for nested entities
 - [Machine Reading Comprehension model](https://github.com/fulstock/mrc_nested_ner_ru)
 - [Second-best Sequence model](https://github.com/fulstock/second-best-learning-and-decoding-rubert)


### Citation
Loukachevitch N., Manandhar S., Baral E., Rozhkov I., Braslavski P., Ivanov V., Batura T., Tutubalina E. NEREL-BIO: a dataset of biomedical abstracts annotated with nested named entities. Bioinformatics. 2023. Volume 39, Issue 4, btad161. https://doi.org/10.1093/bioinformatics/btad161
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
