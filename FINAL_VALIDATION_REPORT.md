# EduGuard — Final Validation Report

**Validation Date:** 2026-11-25  
**Project Directory:** `StudentDropoutPrediction-main/`

---

## 1. Literature Document Verification (Task 1)

- **Total PDFs Discovered:** **30** (`Paper_01_*.pdf` through `Paper_30_*.pdf` in `papers_pdf/`).
- **Text Extraction Status:** **30 / 30** PDFs extracted cleanly via `pypdf` (`0` corrupted, `0` encrypted, `0` image-only).
- **Document Type Classification:** **All 30 / 30 files are 2-page structured research briefs / summaries** (titled `"PAPER #N: RESEARCH BRIEF & METADATA"` on Page 1 with a repository footer on Page 2). **None of the 30 files are complete original full-text research papers.**
- **Relevance Classification:** 26 briefs directly address student dropout prediction (`01–13`, `16–28`); 4 briefs address related educational ML/XAI (`14, 30` career counseling/placement; `15, 29` grade prediction/academic success).
- **Duplicate / Near-Duplicate Pairs:** `(1, 5)`, `(11, 12)`, `(14, 30)`, `(15, 29)`, `(17, 28)`, `(18, 27)`, `(20, 26)`.
- **Missing Metadata:** DOIs/URLs are absent in all 30 PDFs (`Not specified in PDF`) and were not fabricated.
- See [PDF_VERIFICATION_REPORT.md](literature/PDF_VERIFICATION_REPORT.md).

---

## 2. Python Environment & Dependency Compatibility (Task 2)

- **Active Verified Python Environment:** `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe` (**Python 3.12.10**).
- **Virtual Environment:** `StudentDropoutPrediction-main/.venv312` (created and verified; Windows Application Control did **not** block native `.pyd` extensions in Python 3.12.10).
- **Installed & Tested Package Versions (`requirements.txt`):**
  `streamlit==1.67.0`, `pandas==3.1.1`, `numpy==2.5.5`, `scikit-learn==1.9.1`, `xgboost==3.4.1`, `lightgbm==4.7.0`, `imbalanced-learn==0.14.2`, `joblib==1.6.0`, `plotly==7.2.0`, `matplotlib==3.11.2`, `seaborn==0.13.2`, `shap==0.54.1`, `lime==0.2.0.1` (with `scikit-image==0.26.0`), `pypdf==7.1.1`, `openpyxl==3.1.5`, `fpdf2==2.9.0`.
- **Previous Python 3.15.2 Note:** Python 3.15.2 (`.venv`) was blocked by Windows Application Control on `_pcg64.cp315-win_amd64.pyd` and lacked a `scikit-image` wheel; switching to Python 3.12.10 (`.venv312`) resolved both issues cleanly.

---

## 3. Actual Model Evaluation Results & Confusion Matrices (Task 3)

Executed via `verify_all.py` on the UCI dataset (`4,424` rows, `36` original features + `Target`; `0` missing values; `0` duplicates; binary target `Dropout (1) = 1,421`, `Non-Dropout (0) = 3,003`), using a stratified 80/20 split (`885` held-out test records), median imputation fit on `X_train` only, `SMOTE` applied to `X_train` only, and `StandardScaler` fit on resampled `X_train` for Logistic Regression. Models serialized to `models/eduguard_artifacts.joblib`.

| Model | Accuracy | Precision (Dropout=1) | Recall (Dropout=1) | F1-Score (Dropout=1) | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.8712 | 0.8172 | 0.7711 | 0.7935 | 0.9233 | 0.8944 |
| **Random Forest** | 0.8915 | 0.8481 | 0.8063 | 0.8267 | **0.9407** | 0.9114 |
| **XGBoost** | **0.8927** | 0.8487 | **0.8099** | **0.8288** | 0.9366 | **0.9116** |
| **LightGBM** | **0.8927** | **0.8539** | 0.8028 | 0.8276 | 0.9372 | 0.9106 |

### Confusion Matrices (Test Split, `n = 885`, rows = True `[0, 1]`, cols = Pred `[0, 1]`)

- **Logistic Regression:** `[[552, 49], [65, 219]]` (49 FP, 65 FN)
- **Random Forest:** `[[560, 41], [55, 229]]` (41 FP, 55 FN)
- **XGBoost:** `[[560, 41], [54, 230]]` (41 FP, 54 FN — highest Dropout Recall `80.99%` and F1 `0.8288`)
- **LightGBM:** `[[562, 39], [56, 228]]` (39 FP, 56 FN — highest Precision `85.39%`)

---

## 4. SHAP & LIME Testing (Task 4)

- **SHAP Verification:** **Passed** for all 4 models (`global` shape `(50, 42)`, positive-class `Dropout=1` extraction, local contributions, and natural-language driver generation). Top local drivers observed on test instance 0: `Academic_Stress_Index` (Logistic Regression), `Curricular_units_2nd_sem_(approved)` (Random Forest), `Credit_Completion_Rate_2nd_Sem` (XGBoost, LightGBM).
- **LIME Verification:** **Passed** (`10` local rules for `Dropout=1` and raw HTML output generated).

---

## 5. Batch CSV Validation & PDF Generation Testing

- **Batch CSV Validation (`validate_batch_csv`):** **Passed** all 5 edge cases (valid CSV, missing optional column `GDP`, missing required column `Debtor` rejected, extra column ignored cleanly, empty CSV rejected).
- **PDF Report Generation (`early_warning.py`):** **Passed** (`student=2,029 bytes`, `cohort=1,528 bytes`).
- **Research Knowledge Base (`knowledge_base.py`):** **Passed** (`30` papers, `14` categories, `7` preset queries).

---

## 6. Streamlit Runtime Testing (Task 5)

- **Programmatic 7-Page Navigation (`streamlit.testing.v1.AppTest`):** **Passed** with zero exceptions across all 7 pages:
  1. `🏠 Dashboard`
  2. `📊 Data Explorer`
  3. `🤖 Model Hub`
  4. `⚠️ Early Warning System`
  5. `🔮 Student Predictor`
  6. `📈 Reports`
  7. `📚 Research Knowledge Base`
- **Deprecation Note Observed:** Streamlit 1.67.0 logs a deprecation notice for `use_container_width=True` (recommending `width='stretch'` after 2025-12-31); all pages render without error.

---

## 7. Summary of Test Status

| Check / Task | Status | Notes |
|---|---|---|
| Task 1: Literature PDF extraction & document-type verification | **Passed** | 30/30 PDFs extracted; all 30 identified as structured research briefs |
| Task 2: Python 3.12.10 environment & `requirements.txt` install | **Passed** | `.venv312` created; all 16 packages + `scikit-image` installed cleanly |
| Task 3: Live ML training & evaluation (`verify_all.py`) | **Passed** | All 4 models trained, evaluated, and serialized to `models/eduguard_artifacts.joblib` |
| Task 4: Live SHAP & LIME execution | **Passed** | Verified across all 4 models + raw LIME HTML |
| Batch CSV 5-case validation & `fpdf2` PDF exports | **Passed** | All 5 edge cases + student & cohort PDFs verified |
| Task 5: Streamlit 7-page navigation (`AppTest`) | **Passed** | All 7 pages rendered with zero exceptions |
| Interactive browser click-through (manual visual inspection) | **Pending User Run** | Run `.\.venv312\Scripts\streamlit.exe run main.py` locally to inspect in browser |
