# EduGuard — Technical Validation Report

**Date:** 2026-11-25  
**Scope:** Complete technical code review, bug fixes, Python 3.12.10 migration (`.venv312`), dependency locking, and live execution validation for `StudentDropoutPrediction-main/`.

---

## 1. Issues Identified & Corrected During Code Review
1. **3-Class vs. Binary Target Framing:** Corrected to binary `Dropout = 1`, `Enrolled/Graduate = 0` across training, evaluation, SHAP, LIME, and UI.
2. **Pre-Split Data Leakage:** Moved median imputation, `StandardScaler`, and `SMOTE` strictly after the stratified 80/20 `train_test_split` (fit on `X_train` only).
3. **Engineered Indicator Naming & Terminology:** Standardized `Financial_Risk` alongside `Credit_Completion_Rate_1st_Sem`, `Credit_Completion_Rate_2nd_Sem`, `Grade_Drop`, `Engagement_Score`, and `Academic_Stress_Index`; removed "patentable" claims.
4. **Raw LIME HTML Output:** Restored expandable raw LIME HTML (`exp.as_html()` via `components.html`) alongside the Plotly bar chart.
5. **Batch CSV Validation:** Added `validate_batch_csv` to reject empty CSVs and CSVs missing required columns while warning and imputing missing optional columns with `train_medians`.
6. **Dashboard `Intervene` Button:** Restored Priority At-Risk Students rows with `🔍 Intervene` buttons linking directly to `⚠️ Early Warning System`.
7. **FPDF2 Formatting:** Switched `"Arial"` to `"Helvetica"` and added `set_auto_page_break` and `set_x(pdf.l_margin)` / `pdf.epw` to prevent `multi_cell` line-break errors.

---

## 2. Python 3.12 Environment & Dependency Verification
- **Python Interpreter:** `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe` (**Python 3.12.10**).
- **Virtual Environment:** `StudentDropoutPrediction-main/.venv312` (separate from `.venv`).
- **Pinned `requirements.txt` Compatibility:** All 16 pinned packages installed cleanly with zero version conflicts on Python 3.12.10:
  `streamlit==1.67.0`, `pandas==3.1.1`, `numpy==2.5.5`, `scikit-learn==1.9.1`, `xgboost==3.4.1`, `lightgbm==4.7.0`, `imbalanced-learn==0.14.2`, `joblib==1.6.0`, `plotly==7.2.0`, `matplotlib==3.11.2`, `seaborn==0.13.2`, `shap==0.54.1`, `lime==0.2.0.1` (`scikit-image==0.26.0`), `pypdf==7.1.1`, `openpyxl==3.1.5`, `fpdf2==2.9.0`.

---

## 3. Feature Engineering & ML Pipeline Verification (`verify_all.py`)
- **Dataset:** `4,424` rows, `36` original features + `Target` (`Dropout=1,421`, `Non-Dropout=3,003`), `0` missing values, `0` duplicate rows.
- **Stratified Split:** `80%` train (`3,539` rows) / `20%` held-out test (`885` rows); `SMOTE` applied only to `X_train`.
- **Serialization:** Saved to `models/eduguard_artifacts.joblib`.

### Actual Measured Model Evaluation Results (Held-Out Test Split, `n = 885`)

| Model | Accuracy | Precision (`1`) | Recall (`1`) | F1-Score (`1`) | ROC-AUC | PR-AUC | Confusion Matrix `[[TN, FP], [FN, TP]]` |
|---|---:|---:|---:|---:|---:|---:|---|
| **Logistic Regression** | 0.8712 | 0.8172 | 0.7711 | 0.7935 | 0.9233 | 0.8944 | `[[552, 49], [65, 219]]` |
| **Random Forest** | 0.8915 | 0.8481 | 0.8063 | 0.8267 | **0.9407** | 0.9114 | `[[560, 41], [55, 229]]` |
| **XGBoost** | **0.8927** | 0.8487 | **0.8099** | **0.8288** | 0.9366 | **0.9116** | `[[560, 41], [54, 230]]` |
| **LightGBM** | **0.8927** | **0.8539** | 0.8028 | 0.8276 | 0.9372 | 0.9106 | `[[562, 39], [56, 228]]` |

---

## 4. SHAP, LIME & Batch Prediction Verification
- **SHAP:** Verified across all 4 models (`global` shape `(50, 42)` for `Dropout=1`, local contributions, grounded natural-language explanations).
- **LIME:** Verified `10` local rules for `Dropout=1` and raw HTML output.
- **Batch CSV Validation (`validate_batch_csv`):** Passed all 5 edge cases (valid CSV, missing optional `GDP`, missing required `Debtor` rejected, extra column ignored, empty CSV rejected).

---

## 5. UI, Streamlit & PDF Report Testing
- **Streamlit 7-Page Navigation (`AppTest`):** Passed with zero exceptions across all 7 pages (`🏠 Dashboard`, `📊 Data Explorer`, `🤖 Model Hub`, `⚠️ Early Warning System`, `🔮 Student Predictor`, `📈 Reports`, `📚 Research Knowledge Base`).
- **PDF Report Generation (`early_warning.py`):** Passed (`student=2,029 bytes`, `cohort=1,528 bytes`).

---

## 6. Summary of Tests Passed, Failed, and Not Performed

| Check | Status | Notes |
|---|---|---|
| Python 3.12.10 `.venv312` setup & `requirements.txt` | **Passed** | All 16 pinned packages + `scikit-image` installed cleanly |
| `verify_all.py` (ML training, metrics, CM, SHAP, LIME, Batch CSV, PDF) | **Passed** | Exit code `0` on real UCI dataset |
| Research Knowledge Base (`knowledge_base.py`) | **Passed** | 30/30 papers, 14 categories, 7 preset queries |
| Streamlit 7-page `AppTest` navigation | **Passed** | All 7 pages rendered without exception |
| Interactive manual browser visual click-through | **Not Performed** | Automated `AppTest` + HTTP server check used instead |
| Legacy Python 3.15.2 `.venv` native `.pyd` loading | **Failed** | Blocked by Windows Application Control (superseded by `.venv312`) |

---

## 7. Remaining Limitations
- Risk thresholds (`90% / 70% / 40%`) are configurable prototype boundaries for educator decision support, requiring institutional calibration before production deployment.
- Streamlit 1.67.0 emits deprecation notices for `use_container_width=True` (planned for removal after 2025-12-31).
