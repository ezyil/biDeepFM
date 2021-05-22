# biDeepFM
Implementation of the paper: [biDeepFM: A multi-objective deep factorization machine for reciprocal recommendation](https://www.sciencedirect.com/science/article/pii/S2215098621000744).

## Architecture
![Architecture](images/biDeepFM.png "biDeepFM")

## Sample Data Records
applications.csv
|FirmaId|IlanId |AdayId|CvNo  |PostingDateN    |clientJobrefno|
|-------|-------|------|------|----------------|--------------|
|53901  |2007552|721984|926420|2018-04-27 00:32|1235478       |

phoneviews.csv
|AdayId    |CerationDate           |CvNo       |DetailsAsKeyValue|EventType |IlanId |RefNumber|
|----------|-----------------------|-----------|-----------------|----------|-------|---------|
|15638860.0|2018-09-10 12:30:51.100|110254861.0|false            |ViewResume|1938436|         |

jobs.csv
|IlanId |FirmaId|egitimdurumu |cinsiyet|MinTecrube|MaxTecrube|iller        |pozisyonId|pozisyonAdi  |Nitelikler   |IlanAciklama |sektorler|GizliIlan|pozisyonTipi|PozisyonSeviyesi|Lang|IseAlinacakKisiSayisi|Askerlik|ehliyet|
|-------|-------|-------------|--------|----------|----------|-------------|----------|-------------|-------------|-------------|---------|---------|------------|----------------|----|---------------------|--------|-------|
|2000499|113161 |YM##UO##UM...|E       |3.0       |0.0       |İstanbul(A...|6         |Acente Tem...|Sigorta se...|İstanbul v...|035000000|0        |1           |4.0             |1.0 |1                    |B##K    |B      |


candidates.csv
AdayId   |Askerlik|CalismaDurumu|Cinsiyet|CvNo     |EgitimDurumu|Ehliyet|IsdenCikisTarihi|IseBaslamaTarihi|IstecrubesiAciklama|PozisyoIsmi|PozisyonId|Yas|YasadigiSehir|departmanAdi|departmanKodu|universiteAdi|universiteKodu|
|--------|--------|-------------|--------|---------|------------|-------|----------------|----------------|-------------------|-----------|----------|---|-------------|------------|-------------|-------------|--------------|
4574187.0|        |1.0          |1.0     |5814538.0|YM          |1.0    |200702.0        |200607.0        |KAREL SANT...      |Sekreter   |177.0     |30 |Bursa        |İşletme     |7.0          |Anadolu Ün...|2.0           |


## Note
For the implementation of multi-objective models used in the research, [DeepCTR](https://github.com/shenweichen/deepctr) package has been used. All multi-objective models have been added to this package and given under `architectures` directory in this repository.


## Citation

- Yıldırım, Ezgi, Payam Azad, and Şule Gündüz Öğüdücü. "biDeepFM: A multi-objective deep factorization machine for reciprocal recommendation." Engineering Science and Technology, an International Journal (2021).


If you find this code useful in your research, please cite it using the following BibTeX:

```bibtex
@article{yildirim2021bideepfm,
  title={biDeepFM: A multi-objective deep factorization machine for reciprocal recommendation},
  author={Y{\i}ld{\i}r{\i}m, Ezgi and Azad, Payam and {\"O}{\u{g}}{\"u}d{\"u}c{\"u}, {\c{S}}ule G{\"u}nd{\"u}z},
  journal={Engineering Science and Technology, an International Journal},
  year={2021},
  publisher={Elsevier}
}
```
