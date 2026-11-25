# Synthesized Research Findings (30 Papers)

**Project:** EduGuard — Student Dropout Prediction & Early Warning System  
**Scope:** Findings synthesized strictly from the 30 verified PDFs in `papers_pdf/`.

---

## 1. Machine Learning Algorithms & Empirical Performance
- **Tree-Based Ensembles Dominate Tabular Dropout Prediction:** Across systematic reviews (Papers 17, 28 — Lopez-Muñoz et al., 2026), **Random Forest** (61 studies) and **XGBoost** (34 studies) are the two most frequently adopted and top-performing algorithms.
- **Gradient Boosting (XGBoost & LightGBM):**
  - **XGBoost** led in multiple institutional benchmarks: Moroccan higher education (17,883 records, AUC 0.993, Recall 0.944, F1 0.911 — Paper 25), Tec de Monterrey (92% accuracy, AUC 0.960 — Papers 20, 26), asynchronous online learning (89% accuracy, AUC 0.93 — Paper 6), MOOCs/OULAD (87% pass/fail, 86% dropout — Paper 9), Chinese university (88% accuracy, 85% recall — Paper 3), and enrollment-only prediction on ~40,000 Mexican students (88% sensitivity, AUC 0.6902 — Paper 24).
  - **LightGBM** with balanced class weighting and random oversampling achieved strong first-semester results on the **exact UCI 4,424-student dataset** used by EduGuard: **87.1% accuracy, 80.4% precision, 79.2% recall, 79.8% F1, 91.2% AUC-ROC** (Paper 8 — Aliraqi et al., 2026), and 95.26% accuracy in Sebelas Maret (Paper 23).
- **Random Forest & Logistic Regression:**
  - **Random Forest** achieved 96.01% accuracy at Sebelas Maret (Paper 23), 86.6% accuracy / 0.940 AUC on OULAD (Paper 4), and 92% accuracy on grade prediction (Papers 15, 29).
  - **Logistic Regression** serves as a transparent linear baseline (Papers 1, 5, 11, 12, 13, 18, 20, 23, 25, 27), achieving 61.97% sensitivity / 69.51% specificity with SMOTE on 6,617 Colombian students (Papers 11, 12) and 83.68% accuracy at Sebelas Maret (Paper 23).
- **Recall-First Trade-off in EWS:** Paper 4 (Romdhoni et al., 2026) and Paper 24 prioritize **Recall / Sensitivity** because missing an at-risk student (false negative) is costlier than a false alarm in early warning triage.

## 2. Class Imbalance & Evaluation Methodology
- **SMOTE & Resampling:** Papers 2, 8, 9, 11, 12, 20, 23, 26 apply **SMOTE** or class-weighted/oversampled training to counter minority dropout class imbalance.
- **Precision-Recall (PR) vs. ROC Curves:** Paper 2 (Lee & Chung, 2019; 165,715 students) demonstrates that **PR curves (PR-AUC 0.898)** are more informative than ROC curves (ROC-AUC 0.991) under class imbalance.

## 3. Key Predictive Features & Feature Engineering
- **First-Semester Academic Velocity:** Approved vs. enrolled credits, cumulative/semester GPA, and midterm/assignment scores are primary predictors (Papers 3, 8, 15, 23, 25, 29). Paper 8 confirms first-semester data alone is sufficient for reliable early dropout identification.
- **Grade Trajectory (`IPS_range` / Grade Drop):** Semester-to-semester GPA range and drops signal acute academic friction (Papers 6, 23).
- **Behavioral Engagement & Activity Span:** LMS login frequency, assignment submission regularity, and Activity Span (41.3% of predictive power in Paper 4; VLE behavior 85.0% overall).
- **Socioeconomic & Psychosocial Factors:** Financial obligations/income, parental background, family risk, and psychological/stress indicators (Papers 11, 12, 16, 19, 23).

## 4. Explainable AI (XAI) & Counseling Integration
- **SHAP vs. LIME:** Paper 16 (Melo et al., 2022) quantitatively evaluated explainability on 75 features and found **SHAP achieved the highest Explainability Index ($IE_{XAI} = 0.78$)** compared to **LIME ($0.57$)** and Shapley Sampling Values ($0.35$).
- **User-Facing XAI & Counterfactuals:** Paper 10 (Nagy & Molontay, 2023) combined SHAP and LIME with a stakeholder user study. Papers 6, 15, 19, 25, 26, 29 use SHAP (and LIME) to surface case-level risk and protective factors.
- **Actionable Counseling (ADPoCS) & Profiling:** Paper 19 (ADPoCS) links ML + SHAP directly to personalized academic and mental-health counseling; Paper 3 reports 86% GPA improvement among high-risk students receiving counseling/tutoring; Paper 22 (Gallego et al., 2021) segments students into 3 cognitive/learning-strategy profiles via K-Means + PCA.
