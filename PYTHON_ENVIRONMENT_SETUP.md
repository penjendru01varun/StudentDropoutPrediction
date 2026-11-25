# EduGuard — Python 3.12 Environment Setup & Testing Guide

This guide prepares **EduGuard** for execution in a clean **Python 3.12** environment, resolving the Python 3.15 / `scikit-image` wheel gap and Windows Application Control native-extension blocks observed on the local host.

---

## 1. Why Python 3.12?

- **Prebuilt Binary Wheels (`cp312`):** All packages in `requirements.txt`—including `numpy` (`>=1.26.0` for `cp312`), `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `shap`, and `lime` (including its `scikit-image` dependency)—provide official prebuilt wheels for **Python 3.12** without requiring C/Meson source builds.
- **API Compatibility:** All EduGuard modules (`main.py`, `ml_pipeline.py`, `xai_engine.py`, `early_warning.py`, `knowledge_base.py`, `verify_all.py`) were audited against the minimum versions in `requirements.txt` and use standard, non-deprecated APIs (no `use_label_encoder` in XGBoost, `shap.Explanation` for waterfall plots, `fpdf2` `pdf.epw` layout, and Streamlit `>=1.28.0` caching).

### Verified `requirements.txt` Compatibility Reference (Python 3.12)

| Package | `requirements.txt` Spec | Python 3.12 Wheel Note |
|---|---|---|
| `streamlit` | `>=1.28.0` | Compatible (`1.28+` – `1.4x+`) |
| `pandas` | `>=2.0.0` | Compatible (`2.1+` recommended for `cp312`) |
| `numpy` | `>=1.24.0` | `1.26.0+` ships official `cp312` wheels |
| `scikit-learn` | `>=1.3.0` | `1.3.2+` ships official `cp312` wheels |
| `xgboost` | `>=2.0.0` | Compatible (`py3-none-win_amd64` / `manylinux`) |
| `lightgbm` | `>=4.0.0` | Compatible (`py3-none-win_amd64` / `manylinux`) |
| `imbalanced-learn` | `>=0.11.0` | Compatible (`py3-none-any`) |
| `joblib` | `>=1.3.0` | Compatible (`py3-none-any`) |
| `plotly` | `>=5.18.0` | Compatible (`py3-none-any`) |
| `matplotlib` | `>=3.7.0` | `3.8+` ships official `cp312` wheels |
| `seaborn` | `>=0.13.0` | Compatible (`py3-none-any`) |
| `shap` | `>=0.41.0` | `0.44+` ships official `cp312` wheels |
| `lime` | `>=0.2.0` | Compatible (`scikit-image` has `cp312` wheels) |
| `pypdf` | `>=4.0.0` | Pure Python (`py3-none-any`) |
| `openpyxl` | `>=3.1.0` | Pure Python (`py3-none-any`) |
| `fpdf2` | `>=2.7.0` | Pure Python (`py3-none-any`) |

---

## 2. Local Python 3.12 Setup Commands

### Windows (PowerShell)
Navigate into the project folder (`StudentDropoutPrediction-main/`):

```powershell
# 1. Create a clean Python 3.12 virtual environment
py -3.12 -m venv .venv312

# 2. Activate the virtual environment
.\.venv312\Scripts\Activate.ps1

# 3. Upgrade pip and install all dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Run end-to-end ML, XAI, Knowledge Base, and PDF verification
python verify_all.py

# 5. Launch the EduGuard Streamlit dashboard
streamlit run main.py
```

### Linux / macOS (Bash)

```bash
# 1. Create and activate Python 3.12 virtual environment
python3.12 -m venv .venv312
source .venv312/bin/activate

# 2. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Run end-to-end verification
python verify_all.py

# 4. Launch the Streamlit dashboard
streamlit run main.py
```

---

## 3. Alternative Testing Workflow (Approved Cloud Python Environment)

If your local Windows machine continues to enforce **Windows Application Control** policies that block unsigned `.pyd` native extensions (such as `numpy/random/_pcg64*.pyd`), use an approved cloud Linux Python 3.12 environment (without modifying or bypassing local OS security policies):

### Option A: GitHub Codespaces (Recommended for Full Streamlit + `verify_all.py`)
1. Open the repository in **GitHub Codespaces** (Ubuntu Linux container with Python 3.12).
2. In the Codespaces terminal inside `StudentDropoutPrediction-main/`:
   ```bash
   python3 -m venv .venv312
   source .venv312/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   python verify_all.py
   streamlit run main.py --server.port 8501
   ```
3. Codespaces automatically forwards port `8501` so you can interactively test all 7 Streamlit pages in your browser.

### Option B: Google Colab (For `verify_all.py` ML & XAI Verification)
1. Upload or clone `StudentDropoutPrediction-main/` into Colab (Python 3.10–3.12 Linux runtime).
2. Install dependencies and run `verify_all.py`:
   ```bash
   %cd StudentDropoutPrediction-main
   !pip install -q -r requirements.txt
   !python verify_all.py
   ```
3. Record the printed Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Confusion Matrices, and SHAP/LIME verification outputs into `FINAL_VALIDATION_REPORT.md`.
