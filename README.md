# 🎓 EduGuard — Student Dropout Prediction & Early Warning System

**EduGuard** is an Explainable AI (XAI)-based Early Warning System designed to support student retention in higher education institutions.

It uses machine learning to identify students at risk of dropping out, explain individual risk predictions, and provide actionable academic, financial, and engagement-related intervention recommendations.

The system is evaluated using the UCI Student Dropout and Academic Success dataset, containing 4,424 student records and 36 original features. Its research knowledge base contains 30 structured research briefs covering student dropout prediction, educational machine learning, and explainability. These briefs are secondary summaries; original publication sources are subject to separate verification.

---

## 🏗️ Modular Architecture

```text
StudentDropoutPrediction-main/
├── data/
│   └── student_dropout_data.csv      # UCI dataset (4,424 rows, 36 features + Target)
├── papers_pdf/                       # 30 research brief PDFs (all verified)
├── literature/
│   ├── complete_literature_analysis.md
│   ├── literature_matrix.csv
│   ├── literature_index.json
│   ├── research_findings.md
│   ├── research_gaps.md
│   ├── PDF_VERIFICATION_REPORT.md
│   └── papers/paper_01.md ... paper_30.md
├── ml_pipeline.py                    # Leakage-free binary ML pipeline (LR, RF, XGB, LGBM + SMOTE + joblib)
├── xai_engine.py                     # SHAP (global/local/waterfall) & LIME tabular explainability
├── early_warning.py                  # Configurable risk tiers, interventions, PDF dossier generation
├── knowledge_base.py                 # Pure-Python BM25 local RAG over the 30 analyzed papers
├── main.py                           # Dark glassmorphism Streamlit dashboard
└── requirements.txt
```

---

## 🧠 Machine Learning & XAI Methodology

- **Task Framing:** Binary classification (`Dropout = 1`, `Enrolled/Graduate = 0`).
- **Leakage Prevention:** Stratified 80/20 train-test split **before** median imputation, `StandardScaler`, and `SMOTE` (applied only to `X_train`).
- **Engineered Features (Row-Wise):** `Credit_Completion_Rate_1st_Sem`, `Credit_Completion_Rate_2nd_Sem`, `Grade_Drop`, `Engagement_Score`, `Academic_Stress_Index`, `Financial_Risk`.
- **Models Compared:** Logistic Regression, Random Forest, XGBoost, LightGBM (serialized to `models/eduguard_artifacts.joblib`).
- **Explainability:** Global & local SHAP (including waterfall plots) + LIME local rules + grounded natural-language risk explanations.
- **Decision Support:** Configurable prototype thresholds (`Critical >= 90%`, `High 70–89%`, `Medium 40–69%`, `Low < 40%`) mapped to academic, financial, and engagement interventions.

---

## 🚀 How to Run

### Prerequisites
- Python 3.12.10
- pip

### Installation

```powershell
# Activate the tested virtual environment
.\.venv312\Scripts\Activate.ps1

# Install project dependencies if required
pip install -r requirements.txt

# Launch EduGuard
streamlit run main.py
```

Open your browser at:

http://localhost:8501

**Note:** The project has been tested with Python 3.12.10 and its `.venv312` environment.
