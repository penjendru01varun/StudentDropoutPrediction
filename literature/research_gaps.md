# Research Gap Analysis — 30-Paper Literature vs. EduGuard

**Project:** EduGuard — Student Dropout Prediction & Early Warning System  
**Basis:** Verified findings and limitations across the 30 PDFs in `papers_pdf/`.

---

## 1. Commonly Used Prediction Algorithms
Across the 30 reviewed studies (and meta-analyses in Papers 17 & 28 covering 100+ studies):
- **Primary Tabular Models:** Random Forest (61 studies in systematic review), XGBoost (34 studies), LightGBM (Papers 3, 8, 10, 23), Logistic Regression (baseline in Papers 1, 4, 5, 11, 12, 13, 18, 20, 23, 25, 27).
- **Secondary / Specialized Models:** SVM (Papers 4, 13, 14, 18, 23), MLP Neural Networks (Papers 9, 13, 16, 20, 25), k-NN / IBL (Papers 7, 13, 14, 20, 25), Naive Bayes (Papers 9, 14, 20, 25, 30), CatBoost (Paper 10), ICRM rule mining (Paper 21), and K-Means/PCA clustering (Paper 22).

## 2. Commonly Used Educational Datasets
- **Public Benchmarks:** UCI Higher Education Predictors of Student Retention (4,424 records, 36 features — Paper 8), OULAD (32,593 students — Papers 4, 9).
- **Institutional Datasets:** NEIS South Korea (165,715 — Paper 2), Mexican public university (~40,000 — Paper 24), Moroccan higher ed (17,883 — Paper 25), BME Hungary (8,508 — Paper 10), UTP Colombia (6,617 — Papers 11, 12), Sebelas Maret (2,476 — Paper 23), IFRN Brazil (Paper 16), Chinese university (141 — Paper 3).
- **Synthetic / Small Datasets:** Synthetic behavioral/70k datasets (Papers 1, 5, 19) and 215-instance placement data (Papers 14, 30).

## 3. Important Academic & Socioeconomic Risk Factors
- **Academic:** First-semester approved credits, credit completion rate, GPA / semester grades, grade drops (`IPS_range`), assignment regularity, and pre-enrollment admission grades (Papers 3, 6, 8, 15, 23, 24, 25).
- **Behavioral / Engagement:** LMS login frequency, learning inactivity, activity span, evaluations completed (Papers 4, 6).
- **Socioeconomic & Psychosocial:** Tuition delinquency, debtor status, lack of scholarship/financial aid, family income, commuting/displacement, academic stress, and mental health/wellbeing (Papers 11, 12, 16, 19, 23).

## 4. Identified Limitations & Research Gaps in Reviewed Literature
1. **Lack of Operational Real-Time Early Warning Systems (Papers 17, 28):** Systematic reviews explicitly note that most published studies stop at offline model training and lack an operational, educator-facing early warning system.
2. **Disconnect Between Prediction, Explainability, and Intervention (Papers 17, 19, 28):** Many high-accuracy models remain black boxes (noted in Papers 5, 15, 29), or provide SHAP/LIME plots without translating risk drivers into structured, multi-disciplinary counseling workflows.
3. **Lack of Standardized Interpretability Metrics & LIME Scalability (Papers 16, 29):** Paper 29 highlights LIME scalability issues on large datasets and the absence of standardized interpretability metrics, while Paper 16 demonstrates SHAP's superior $IE_{XAI}$ (0.78 vs. 0.57 for LIME).
4. **Inconsistent Class Imbalance & Leakage-Safe Evaluation:** While Papers 2, 8, 9, 11, 12, 20, 23, 26 use SMOTE or resampling, many studies rely solely on Accuracy rather than Recall, F1, ROC-AUC, and PR-AUC (Paper 2).
5. **Single-Institution / Synthetic Generalizability Limits (Papers 1, 3, 5, 19, 24):** Several studies use synthetic data or single-institution cohorts, and Paper 24 shows enrollment-only features yield lower AUC (0.6902) than first-semester academic features (Paper 8: 0.912 AUC).

---

## 5. Comparison: Existing Literature Approaches vs. EduGuard

| Dimension | Typical Existing Studies (30-Paper Corpus) | EduGuard Implementation |
|---|---|---|
| **Scope & Deployment** | Offline notebooks/experiments; lack operational EWS (Papers 17, 28) | End-to-end interactive Streamlit EWS dashboard with individual, batch CSV, and PDF dossier workflows |
| **ML Pipeline & Leakage Control** | Often unspecified split/SMOTE ordering; mixed binary vs. multi-class | Binary Dropout (`1`) vs. Non-Dropout (`0`) on UCI 4,424 dataset; stratified 80/20 split *before* SMOTE & scaling; `joblib` serialization |
| **Model Comparison** | 1–4 models on varying splits | Fair comparison of **Logistic Regression, Random Forest, XGBoost, LightGBM** on the identical stratified test split (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix) |
| **Feature Engineering** | Raw institutional fields or LMS clickstreams only | 5+ domain-grounded engineered features (`Credit_Completion_Rate_1st_Sem`, `Credit_Completion_Rate_2nd_Sem`, `Grade_Drop`, `Engagement_Score`, `Academic_Stress_Index`, `Financial_Risk_Score`) with zero-division guards |
| **Explainable AI (XAI)** | Either none (black box) or SHAP/LIME plots without plain-language synthesis | Dual **SHAP** (global summary + local waterfall/contributions) & **LIME** (local rules) + grounded natural-language risk explanations |
| **Intervention & Decision Support** | Prediction score only (except ADPoCS Paper 19 & Paper 3) | Configurable prototype risk tiers (Critical ≥90%, High 70–89%, Medium 40–69%, Low <40%) mapped to academic, financial, and engagement decision-support recommendations |
| **Literature Traceability** | Static bibliography | Built-in searchable Research Knowledge Base indexing all 30 verified papers with page references |

> [!IMPORTANT]
> EduGuard builds upon established methods from the reviewed literature (e.g., SMOTE from Paper 2, LightGBM/UCI framing from Paper 8, dual SHAP+LIME from Papers 10, 15, 16, 29, and counseling integration from Paper 19) rather than claiming to be the first system to use these individual techniques. Its contribution lies in integrating leakage-free modeling, dual XAI, configurable risk triage, and literature-grounded decision support in one cohesive platform.
