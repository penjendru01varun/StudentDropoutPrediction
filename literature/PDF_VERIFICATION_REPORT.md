# PDF Verification Report — `papers_pdf/`

**Verification Date:** 2026-11-25  
**Extraction Tool:** `pypdf` (`PdfReader`) executed on every `.pdf` file in `StudentDropoutPrediction-main/papers_pdf/`.

---

## 1. Document Type vs. Text Extraction Status

> [!IMPORTANT]
> **Document Type Finding:** All **30 / 30** PDFs in `papers_pdf/` are **2-page structured research briefs / summaries** (generated from `research_data.py` via `generate_paper_pdfs.py`, titled `"PAPER #N: RESEARCH BRIEF & METADATA"` on Page 1 with a repository footer on Page 2). **None of the 30 files are complete original full-text research papers.**

| Verification Dimension | Count | Status / Details |
|---|---|---|
| **Total PDFs discovered** | **30** | `Paper_01_*.pdf` through `Paper_30_*.pdf` |
| **Total PDFs successfully extracted** | **30** | All 30 opened cleanly via `pypdf` (2 pages each, 1,313–2,085 extracted characters) |
| **Complete original research papers** | **0** | None are full-text original journal/conference PDFs |
| **Structured research briefs / summaries** | **30** | All 30 contain Title, Authors, Publication/Year, Dataset, Algorithms, Abstract, Key Findings, Top Model, EduGuard Blueprint, and Reported Metrics |
| **Abstracts / partial papers / other** | **0** | — |
| **Failed or unreadable PDFs** | **0** | None corrupted, encrypted, or image-only; OCR not required |

---

## 2. Missing Metadata & Source Verification Notes

Across all 30 structured research briefs:
- **DOI / Publication URL:** Missing in **all 30 PDFs** (`Not specified in PDF`).
- **Explicit Dataset Size / Feature Count:** Not explicitly stated in PDF text for Papers 1, 5, 6, 7, 13, 15, 18, 19, 20, 21, 22, 26, 27, 29.
- **Explicit Limitations / Future Work Section:** Not separately titled in the 2-page brief format (limitations are only explicitly mentioned in abstract/findings for Papers 5, 17, 28, 29).
- **Page References:** All substantive content is on **Page 1** (`[p. 1]`); Page 2 contains only the repository footer.
- **Original Source Cross-Check:** Because the files in `papers_pdf/` are locally synthesized research briefs rather than publisher PDFs, full methodology tables, hyperparameter settings, and DOIs could not be verified from the PDFs themselves and are not fabricated.

---

## 3. Duplicate / Near-Duplicate Papers Identified

| Pair | Filenames | Relationship |
|---|---|---|
| **Papers 1 & 5** | `Paper_01_Random_Forest.pdf` & `Paper_05_Random_Forest.pdf` | Same title & authors (Zhao & Ren, 2025, Bulletin of Education and Psychology; Paper 5 adds `(HAL)` and notes black-box limitation). |
| **Papers 11 & 12** | `Paper_11_Logistic_Regression_with_SMOTE.pdf` & `Paper_12_Logistic_Regression_with_SMOTE.pdf` | Same authors (Hoyos Osorio & Daza Santacoloma, 2023) & dataset (UTP Colombia, 6,617 students); Paper 12 is marked `(Extended)` and adds specificity (69.51%) & 20-feature selection. |
| **Papers 14 & 30** | `Paper_14_Naive_Bayes_(Recall).pdf` & `Paper_30_Naive_Bayes_(Recall_91.2%).pdf` | Same title, authors (Guleria & Sood), dataset (215 instances, 15 attrs), and metrics (91.2% Recall, 90.7% F-Measure); Paper 14 lists 2023, Paper 30 lists 2022. |
| **Papers 15 & 29** | `Paper_15_Random_Forest_+_SHAP_LIME.pdf` & `Paper_29_Random_Forest_+_SHAP_LIME.pdf` | Same title, year (2025), journal, Omani dataset, and metrics (92% Acc, 0.90 F1); Paper 15 lists E. Ben George et al., Paper 29 lists Emma S. George and notes LIME scalability limitations. |
| **Papers 17 & 28** | `Paper_17_Random_Forest_(61_studies)___XGBoost_(34_studies).pdf` & `Paper_28_Random_Forest_&_XGBoost.pdf` | Same systematic review by Lopez-Muñoz et al. (2026); Paper 28 is labeled `(Review 2)`. |
| **Papers 18 & 27** | `Paper_18_Gradient_Boosting___Random_Forest.pdf` & `Paper_27_Gradient_Boosting.pdf` | Near-identical abstract/methods on Multi-institutional Higher Ed Dataset (2024; Journal of Academic Analytics vs. Heliyon). |
| **Papers 20 & 26** | `Paper_20_XGBoost_(92%_Accuracy).pdf` & `Paper_26_XGBoost_&_Random_Forest_(92%_Accuracy).pdf` | Same 10-algorithm study on Tecnologico de Monterrey (2023; Computers & Education vs. Applied Soft Computing; Paper 26 emphasizes SHAP). |

---

## 4. Relevance Classification

- **Directly Related to Student Dropout Prediction (26 briefs):**
  Papers `01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28`.
- **Related Educational ML & Explainability (4 briefs):**
  - Career Counseling / Placement Prediction: Papers `14, 30`.
  - Student Grade Prediction / Academic Success: Papers `15, 29`.
