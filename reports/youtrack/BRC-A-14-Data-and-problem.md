<!-- Update for BRC-A-14 "Data and Problem description" — add the two sections below at the end of the article. The existing goals / NHG / raw-data sections stay valid. -->

## Label semantics (2026-11, corrected)

The cohort is rebuilt by [`scripts/make_cohort.py`](https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/breast-cancer-grading/-/blob/master/scripts/make_cohort.py)
from the MOU Excel and lives at `/mnt/projects/breast_cancer/brc_grading/labels.csv` (1735 slides,
columns `slide_id, grade, split`):

* `grade` **0 = no tumour epithelium** (an `-N` slide), **1/2/3 = Nottingham G1/G2/G3**
* `-T` slides carry the case's Nottingham grade, `-N` slides are class 0
* filtered out: grade 9 (not assessable) and neoadjuvant cases
* an older note calling the classes "B-RAD grades 2–5" was **wrong** — counts were unaffected, only
  the names attached to them

## Label defects found by expert review (2026-11-14)

The pathologist reviewed the 8 worst test errors ([`reports/zprava_pro_patologa_komentare.md`](https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/breast-cancer-grading/-/blob/master/reports/zprava_pro_patologa_komentare.md)).
Verdicts: 3 outright label errors (the model was right), 3 defensible label-semantics cases,
1 genuine borderline, only **1 real model mistake**. Concrete data defects to fix in a cohort v2:

| defect | example | note |
|---|---|---|
| multi-tumour grade assignment | `2021_02976-08-T1` | a breast with G1+G2+G3 tumours; the slide is the G3 one but carries the G1 label |
| morphological code mismatch (8500 vs 8480) | `2021_00353-21-T2` | mucinous carcinoma (8480) labelled as the G3 of an 8500 ductal; should be G2 |
| **in-situ lesions (DCIS) inside class N** | `2021_01085-02-N` | labelled N but contains DCIS — the model called it G3 and was right; the N class is not "tumour-free" by this definition |
| Nottingham score vs grade inconsistency | `2020_00168-02-T` | score 7 = G2, labelled G3 (undercounted mitoses) |

Implications: the reported accuracies are **lower bounds** (part of the measured "model error" is
reference-standard noise). A v2 cohort must re-grade **blind to the model's predictions** — fixing only
the slides the model got wrong would contaminate the test set.
