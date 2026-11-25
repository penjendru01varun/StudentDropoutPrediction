# Complete Literature Analysis — 30 Research Papers

**Project:** EduGuard — Student Dropout Prediction & Early Warning System  
**Corpus Verified:** 30 / 30 PDF files in `papers_pdf/` (all 2-page structured research briefs, extracted and verified via `pypdf`).

> [!NOTE]
> Every finding, metric, dataset, and limitation below is extracted directly from the actual PDF contents in `papers_pdf/`. Fields not present in the PDFs (such as DOIs/URLs or full-paper hyperparameter tables) are explicitly marked as *Not specified in PDF* rather than fabricated.

---

## 1. Corpus Overview & Thematic Breakdown

- **Directly Related to Student Dropout Prediction (26 papers):** Papers 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28.
- **Related Educational ML & Explainability Studies (4 papers):**
  - *Grade Prediction & Academic Success:* Papers 15 & 29 (Omani higher education dataset, Random Forest + SHAP/LIME).
  - *Career Counseling & Placement:* Papers 14 & 30 (Kaggle educational placement dataset, 215 instances, 15 attributes, Naive Bayes + XAI).

---

## 2. Master Summary Table (All 30 Papers)

| # | Filename | Title (Short) | Authors (Year) | Dataset | Best Algorithm | Key Reported Metrics | XAI |
|---|---|---|---|---|---|---|---|
| 1 | `Paper_01_Random_Forest.pdf` | RF-Based EWS Using Behavioral Data | Zhao & Ren (2025) | Synthetic behavioral data | Random Forest | Acc: 1.00, AUC: 1.00, Rec: 1.00 | None |
| 2 | `Paper_02_Boosted_Decision_Tree_with_SMOTE.pdf` | ML Dropout EWS with SMOTE & PR Curves | Lee & Chung (2019) | NEIS South Korea (165,715 students) | Boosted Decision Tree + SMOTE | Acc: 0.96, ROC-AUC: 0.991, PR-AUC: 0.898 | None |
| 3 | `Paper_03_XGBoost.pdf` | ML in Higher Ed EWS & Intervention | Yonglin Zhao (2025) | Chinese public university (141 BS students) | XGBoost | Acc: 0.88, Prec: 0.81, Rec: 0.85, F1: 0.83, AUC: 0.92 | None |
| 4 | `Paper_04_SVM_(Recall)___Random_Forest_(Accuracy).pdf` | EWS from LMS Behavioral Engagement | Romdhoni et al. (2026) | OULAD (32,593 students, 13 features) | SVM (Recall) / RF (Acc) | Acc: 0.866, Rec: 0.903, AUC: 0.940 | Feature importance (VLE 85%, Activity Span 41.3%) |
| 5 | `Paper_05_Random_Forest.pdf` | RF-Based EWS Using Behavioral Data (HAL) | Zhao & Ren (2025) | Synthetic behavioral dataset | Random Forest | Acc: 1.00, AUC: 1.00 | None (notes black-box limitation) |
| 6 | `Paper_06_XGBoost.pdf` | LA-Based EWS in Asynchronous Online Learning | Juliansyah et al. (2026) | LMS behavioral data | XGBoost | Acc: 0.89, F1: 0.87, AUC: 0.93 | SHAP |
| 7 | `Paper_07_k-Nearest_Neighbors_(k-NN___IBL).pdf` | PDE Computational System Proposal | Nicoletti et al. (2020) | Higher ed institutional data | k-NN / IBL | Acc: 0.82, Rec: 0.80 | None |
| 8 | `Paper_08_LightGBM_(Balanced_+_Oversampling).pdf` | First-Semester Dropout Risk Stratification | Aliraqi et al. (2026) | UCI Retention Dataset (4,424 students) | LightGBM (Balanced + Oversampling) | Acc: 0.871, Prec: 0.804, Rec: 0.792, F1: 0.798, AUC: 0.912 | None |
| 9 | `Paper_09_XGBoost.pdf` | Predictive Modeling of Dropout in MOOCs | Patel & Amin (2024) | OULAD | XGBoost | Acc: 0.870 (pass/fail), 0.86 (dropout), AUC: 0.915 | None |
| 10 | `Paper_10_CatBoost___XGBoost.pdf` | Interpretable Dropout Prediction (XAI) | Nagy & Molontay (2023) | BME Hungary (8,508 students) | CatBoost / XGBoost | AUC: 0.774, Avg Prec: 0.847 | SHAP, LIME, user study |
| 11 | `Paper_11_Logistic_Regression_with_SMOTE.pdf` | Predictive Model for College Dropout | Hoyos Osorio & Daza Santacoloma (2023) | UTP Colombia (6,617 students) | Logistic Regression + SMOTE | Sens: 0.6197, Acc: 0.6226 | Linear weights |
| 12 | `Paper_12_Logistic_Regression_with_SMOTE.pdf` | Predictive Model for College Dropout (Extended) | Hoyos Osorio & Daza Santacoloma (2023) | UTP Colombia (6,617 students, 20 features) | Logistic Regression + SMOTE | Sens: 0.6197, Spec: 0.6951 | Linear weights |
| 13 | `Paper_13_Ensemble_Gradient_Boosting_&_Random_Forest.pdf` | University Student Dropout Prediction | Higher Ed Analytics Consortium (2024) | University student dataset | Ensemble GB & RF | Acc: 0.885, AUC: 0.932 | None |
| 14 | `Paper_14_Naive_Bayes_(Recall).pdf` | XAI & ML for EDM Career Counseling | Guleria & Sood (2023) | Placement dataset (215 instances, 15 attrs) | Naive Bayes (Recall) | Rec: 0.912, F-Measure: 0.907 | White-box vs Black-box XAI |
| 15 | `Paper_15_Random_Forest_+_SHAP_LIME.pdf` | XAI for Predicting Student Grades | E. Ben George et al. (2025) | Higher ed institution in Oman | Random Forest + SHAP/LIME | Acc: 0.920, Weighted F1: 0.900 | SHAP, LIME |
| 16 | `Paper_16_MLP_+_SHAP.pdf` | XAI to Evaluate School Dropout | Melo et al. (2022) | IFRN Brazil (75 features) | MLP + SHAP | Acc: 0.970, IE_XAI (SHAP): 0.78, (LIME): 0.57 | SHAP, LIME, Shapley Values |
| 17 | `Paper_17_Random_Forest_(61_studies)___XGBoost_(34_studies).pdf` | Systematic Review of ML & Risk Factors | Lopez-Muñoz et al. (2026) | Meta-analysis (100+ studies) | RF (61 studies) / XGBoost (34 studies) | RF: 61 studies, XGB: 34 studies | Identifies XAI-intervention gap |
| 18 | `Paper_18_Gradient_Boosting___Random_Forest.pdf` | Student Dropout Prediction & EWS | Retention Research Group (2024) | Multi-institutional Higher Ed | Gradient Boosting / RF | Acc: 0.880, AUC: 0.925 | RF feature importance |
| 19 | `Paper_19_Random_Forest___XGBoost.pdf` | ADPoCS Prediction & Counselling System | Intelligent Ed. Systems Group (2024) | Synthetic (70,000 records) | Random Forest / XGBoost | Acc: 0.915, AUC: 0.950 | SHAP |
| 20 | `Paper_20_XGBoost_(92%_Accuracy).pdf` | Dropout Prediction at Tec de Monterrey | Tec de Monterrey Analytics Lab (2023) | Tec de Monterrey, Mexico | XGBoost | Acc: 0.920, AUC: 0.960 | None |
| 21 | `Paper_21_ICRM_(Interpretable_Classification_Rule_Mining).pdf` | Early Dropout Prediction (High School) | Marquez-Vera et al. (2016) | High school (Zacatecas, Mexico) | ICRM | Acc: 0.850, Rec: 0.830 | ICRM IF-THEN rules |
| 22 | `Paper_22_K-Means_3-Cluster_Cognitive_Profiling.pdf` | Identifying At-Risk Students via Clustering | Gallego et al. (2021) | Higher education student data | K-Means (3 clusters) | Silhouette: 0.45, 3 profiles | Unsupervised profiling |
| 23 | `Paper_23_Random_Forest_(96.01%_Accuracy).pdf` | Predicting University Dropout Using ML | Sebelas Maret Team (2023) | Sebelas Maret Univ. (2,476 students) | Random Forest | Acc: 0.9601, LGBM: 0.9526, DT: 0.9124 | RFE-CV feature ranking |
| 24 | `Paper_24_XGBoost_(88%_Sensitivity).pdf` | EWS for Student Dropout Using XGBoost | Mexican Retention Consortium (2024) | Mexican public univ. (~40,000 students) | XGBoost | AUC: 0.6902, F1: 0.6946, Sens: 0.8800 | None |
| 25 | `Paper_25_XGBoost_(AUC_0.993).pdf` | Modular Dropout Pipeline with XAI | Moroccan Higher Ed Data Lab (2024) | Moroccan higher ed (17,883 records) | XGBoost | AUC: 0.993, F1: 0.911, Rec: 0.944 | SHAP |
| 26 | `Paper_26_XGBoost_&_Random_Forest_(92%_Accuracy).pdf` | Dropout Prediction with SHAP | Educational Data Science Group (2023) | Tec de Monterrey, Mexico | XGBoost & RF | Acc: 0.920, AUC: 0.955 | SHAP |
| 27 | `Paper_27_Gradient_Boosting.pdf` | Comparison of ML Models for Dropout | Academic Analytics Research Group (2024) | Multi-institutional Higher Ed | Gradient Boosting | Acc: 0.875, AUC: 0.920 | RF feature importance |
| 28 | `Paper_28_Random_Forest_&_XGBoost.pdf` | Systematic Review of ML & Risk Factors (2) | Lopez-Muñoz et al. (2026) | Meta-analysis of Higher Ed studies | RF & XGBoost | RF: #1 (61), XGB: #2 (34) | Identifies XAI-intervention gap |
| 29 | `Paper_29_Random_Forest_+_SHAP_LIME.pdf` | XAI for Predicting Student Grades | Emma S. George (2025) | Omani higher ed dataset | Random Forest + SHAP/LIME | Acc: 0.920, Weighted F1: 0.900 | SHAP, LIME |
| 30 | `Paper_30_Naive_Bayes_(Recall_91.2%).pdf` | XAI & ML for EDM Career Counseling | Guleria & Sood (2022) | Kaggle placement (215 instances, 15 attrs) | Naive Bayes | Rec: 0.912, F-Measure: 0.907 | White-box vs Black-box XAI |

---

## 3. Detailed Individual Analyses

Full 20-point extractions with page references for every paper are stored in [`literature/papers/paper_01.md`](papers/paper_01.md) through [`literature/papers/paper_30.md`](papers/paper_30.md), and the structured tabular matrix is in [`literature/literature_matrix.csv`](literature_matrix.csv).
