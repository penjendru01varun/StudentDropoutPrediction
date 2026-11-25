<<<<<<< HEAD
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
=======
# 🎓 Student Dropout Prediction & Early Warning System (EduGuard)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff69b4.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![SHAP](https://img.shields.io/badge/SHAP-0.41+-violet.svg)
![LIME](https://img.shields.io/badge/LIME-0.2.0-yellowgreen.svg)

A Machine Learning-powered Early Warning System (EWS) that predicts student dropout risk in real-time, provides dual-engine explainable AI (SHAP & LIME) explanations, generates personalized academic and financial interventions, and displays them on a premium glass-morphic dark theme dashboard.

---

## 🎨 Theme & UI/UX Design

- **Theme**: Premium Dark Theme with Glass-morphism effects.
- **Primary Gradient**: `#667eea` → `#764ba2` (Purple-Blue)
- **Secondary Accent**: `#f093fb` → `#f5576c` (Pink)
- **Background**: `#0f0c29` → `#302b63` → `#24243e` (Deep Space gradient)
- **Card Style**: Glass-morphic borders with backdrop-blur (`15px`) and soft glowing shadows.
- **Typography**: Inter / Poppins loaded via Google Fonts.
- **Micro-interactions**: Pulse animations for Critical Risk badges, button click scaling, card hover translations, and styled scrollbars.

---

## 🚀 Key Innovations

### 1. Continuous Triage & Early Warning
Stratifies students in real-time into four risk tiers based on dropout probabilities:
- **Critical (>=90%)** - Triggers a warning pulse animation.
- **High (70-89%)** - Flags immediate mediation.
- **Medium (40-69%)** - Identifies warning signs.
- **Low (<40%)** - Normal monitoring status.

### 2. Novel Engineered Features (Patentable)
- **Credit Completion Rate (1st & 2nd Sem)**: Approved vs. Enrolled units ratio.
- **Grade Drop**: Sem 1 - Sem 2 average grade differential.
- **Engagement Score**: Average ratio of evaluations to enrolled credits.
- **Academic Stress Index**: Enrolled credits minus approved credits (overburden metric).
- **Financial Risk Score**: Combined index of Debtor, Tuition status, and Scholarship status.

### 3. Dual-Engine Explainable AI (XAI)
- **LIME Explanations**: Local decision rule horizontal bar charts explaining predictions.
- **SHAP Explanations**: Global beeswarm summaries, global absolute importances, and local waterfall attribution plots.
- **Natural Language Risk Drivers**: Explains the exact triggers in plain English.

### 4. Automated Intervention Recommendations
Generates priority academic, financial, or socio-demographic intervention checklists linked directly to risk drivers. Includes printable PDF Dossier exports.

---

## 📂 Project Structure

```
├── data/
│   └── student_dropout_data.csv    # UCI Student Dropout Dataset
├── main.py                         # Single-file production-ready Streamlit app
└── requirements.txt                # System dependencies
>>>>>>> 990e08ca4dd488a580c14221d8affe49fa487d86
```

---

<<<<<<< HEAD
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
=======
## 🚀 Run Instructions

### Prerequisites
- Python 3.9+ (Miniconda recommended)

### Installation

1. **Activate your environment or locate your python executable.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

1. **Start the Streamlit application:**
   ```bash
   streamlit run main.py
   ```
2. **Open in browser:**
   Navigate to the local URL (usually `http://localhost:8501`).

*Note: If you run into path/python execution issues on Windows, execute the command directly using your Miniconda path:*
```bash
C:\Users\varun\miniconda3\python.exe -m pip install -r requirements.txt
C:\Users\varun\miniconda3\Scripts\streamlit.exe run main.py
```

---

## 📋 Dashboard Modules

### 1. 🏠 Dashboard (Landing Page)
- Metric cards showing Total Students, Dropout Rates, Critical Risks, and Graduation success.
- Live Pie chart of Risk Distributions.
- Interactive list of High-Risk students requiring immediate mediation, with a "🔍 Intervene" button that links directly to their dossier.

### 2. 📊 Data Explorer
- Demographic distributions (Age, Gender, Scholarship).
- Academic trends (1st sem grades, Grade Drop boxplots).
- Co-factors analysis (Debtors, Academic Stress).
- Interactive parameter plotter.

### 3. 🤖 Model Hub
- Multi-model evaluation table (Accuracy, Precision, Recall, F1, ROC-AUC) comparing Random Forest, XGBoost, LightGBM, and Logistic Regression (trained using SMOTE).
- Interactive Confusion Matrix heatmaps.
- Global SHAP beeswarm and feature importance plots.

### 4. ⚠️ Early Warning System
- Dynamic student risk directory table with color-coded risk level rows.
- Filter and search panel (search by Student ID, filter by Course, Risk, Gender, or Scholarship).
- Live AI explanation and intervention recommendations with downloadable PDF dossiers.
- Batch CSV upload support for bulk predictions.

### 5. 🔮 Predictor
- Interactive input sliders and check buttons for custom student data.
- Gauge indicator showing probability with color-coded risk markers.
- Side-by-side SHAP & LIME charts representing feature-level decision rules.

### 6. 📈 Reports
- Printable PDF Dossier generator.
- Quick copies of the Draft Patent Abstract and Proposed IEEE Paper Outline.
>>>>>>> 990e08ca4dd488a580c14221d8affe49fa487d86
