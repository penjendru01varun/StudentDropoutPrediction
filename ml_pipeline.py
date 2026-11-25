"""
EduGuard Machine Learning Pipeline (Phase 5).

Implements leakage-free binary dropout classification (Dropout = 1,
Enrolled/Graduate = 0) on the UCI Student Dropout and Academic Success dataset
(4,424 records, 36 original features), with:
- Data quality inspection (missing values, duplicates, class distribution)
- Domain-grounded engineered features with zero-denominator guards
- Stratified 80/20 train-test split BEFORE imputation, scaling, and SMOTE
- Fair comparison of 4 core models: Logistic Regression, Random Forest,
  XGBoost, and LightGBM on the identical test split
- Joblib artifact serialization
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BINARY_TARGET_MAP = {"Dropout": 1, "Graduate": 0, "Enrolled": 0}
SCALED_MODELS = {"Logistic Regression"}
CORE_MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]

ENGINEERED_FEATURE_SPECS: List[Dict[str, str]] = [
    {
        "name": "Credit_Completion_Rate_1st_Sem",
        "formula": "approved_1st / enrolled_1st (0.0 if enrolled_1st == 0)",
        "source_columns": "Curricular units 1st sem (approved), Curricular units 1st sem (enrolled)",
        "relevance": "First-semester credit velocity is a primary retention indicator (Papers 3, 8, 17).",
        "assumptions": "Zero enrolled units yields 0.0 completion rate to avoid division by zero.",
    },
    {
        "name": "Credit_Completion_Rate_2nd_Sem",
        "formula": "approved_2nd / enrolled_2nd (0.0 if enrolled_2nd == 0)",
        "source_columns": "Curricular units 2nd sem (approved), Curricular units 2nd sem (enrolled)",
        "relevance": "Measures sustained academic progression into the second semester (Papers 1, 8).",
        "assumptions": "Zero enrolled units yields 0.0 completion rate.",
    },
    {
        "name": "Grade_Drop",
        "formula": "grade_1st_sem - grade_2nd_sem",
        "source_columns": "Curricular units 1st sem (grade), Curricular units 2nd sem (grade)",
        "relevance": "Positive values capture semester-to-semester GPA/grade decline (Papers 6, 23).",
        "assumptions": "Both semester grades are on the same 0-20 institutional scale.",
    },
    {
        "name": "Engagement_Score",
        "formula": "0.5 * (eval_1st / max(enrolled_1st, 1) + eval_2nd / max(enrolled_2nd, 1))",
        "source_columns": "Curricular units 1st/2nd sem (evaluations), Curricular units 1st/2nd sem (enrolled)",
        "relevance": "Approximates assessment participation intensity relative to course load (Papers 4, 6).",
        "assumptions": "Denominator clamped at 1 when enrolled == 0 (yields 0 if evaluations == 0).",
    },
    {
        "name": "Academic_Stress_Index",
        "formula": "(enrolled_1st + enrolled_2nd) - (approved_1st + approved_2nd)",
        "source_columns": "Curricular units 1st/2nd sem (enrolled), Curricular units 1st/2nd sem (approved)",
        "relevance": "Quantifies unapproved credit backlog / academic overburden (Papers 1, 8).",
        "assumptions": "Higher positive values indicate greater uncompleted credit load.",
    },
    {
        "name": "Financial_Risk",
        "formula": "Debtor + (1 - Tuition_fees_up_to_date) + (1 - Scholarship_holder)",
        "source_columns": "Debtor, Tuition fees up to date, Scholarship holder",
        "relevance": "Composite socioeconomic vulnerability indicator (0 to 3) (Papers 11, 16, 19).",
        "assumptions": "Binary indicators (0/1); higher score indicates greater financial pressure.",
    },
]

REQUIRED_RAW_COLUMNS = [
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (grade)",
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for LightGBM/XGBoost compatibility."""
    out = df.copy()
    out.columns = [
        c.replace("\ufeff", "")
        .replace("\t", "")
        .replace("[", "")
        .replace("]", "")
        .replace("<", "")
        .strip()
        .replace(" ", "_")
        .strip("_")
        for c in out.columns
    ]
    return out


def resolve_dataset_path(base_dir: Optional[str] = None) -> Optional[str]:
    """Locate the UCI dataset CSV in standard project directories."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "data", "student_dropout_data.csv"),
        os.path.join(base_dir, "dataset.csv"),
        os.path.join(base_dir, "student_dropout_data.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_raw_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """Load raw CSV with automatic delimiter detection and header cleanup."""
    if path is None:
        path = resolve_dataset_path()
    if path is None or not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "").str.strip()
    return df


def inspect_dataset_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Return missing values, duplicates, shape, and target distribution."""
    if df.empty:
        return {}
    missing_by_col = df.isna().sum()
    target_counts = df["Target"].value_counts().to_dict() if "Target" in df.columns else {}
    binary_counts = {}
    if "Target" in df.columns:
        y_bin = df["Target"].astype(str).map(BINARY_TARGET_MAP)
        binary_counts = {
            "Dropout (1)": int((y_bin == 1).sum()),
            "Non-Dropout (0)": int((y_bin == 0).sum()),
        }
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "features": int(df.shape[1] - (1 if "Target" in df.columns else 0)),
        "total_missing": int(missing_by_col.sum()),
        "missing_columns": missing_by_col[missing_by_col > 0].to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "original_target_counts": target_counts,
        "binary_target_counts": binary_counts,
    }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute row-wise engineered features using verified UCI dataset columns.
    Row-wise computations introduce zero cross-sample data leakage.
    """
    out = df.copy()
    out.columns = out.columns.str.replace("\ufeff", "").str.strip()

    def _col(name: str, default: float = 0.0) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").fillna(default)
        alt = name.replace(" ", "_")
        if alt in out.columns:
            return pd.to_numeric(out[alt], errors="coerce").fillna(default)
        return pd.Series(default, index=out.index, dtype=float)

    enr1 = _col("Curricular units 1st sem (enrolled)", 0.0)
    app1 = _col("Curricular units 1st sem (approved)", 0.0)
    eval1 = _col("Curricular units 1st sem (evaluations)", 0.0)
    grd1 = _col("Curricular units 1st sem (grade)", 0.0)

    enr2 = _col("Curricular units 2nd sem (enrolled)", 0.0)
    app2 = _col("Curricular units 2nd sem (approved)", 0.0)
    eval2 = _col("Curricular units 2nd sem (evaluations)", 0.0)
    grd2 = _col("Curricular units 2nd sem (grade)", 0.0)

    # 1. Credit Completion Rate (1st & 2nd Sem) with zero-denominator guard
    out["Credit_Completion_Rate_1st_Sem"] = np.where(enr1 > 0, app1 / enr1, 0.0)
    out["Credit_Completion_Rate_2nd_Sem"] = np.where(enr2 > 0, app2 / enr2, 0.0)

    # 2. Grade Drop
    out["Grade_Drop"] = grd1 - grd2

    # 3. Engagement Score
    eng1 = np.where(enr1 > 0, eval1 / enr1, 0.0)
    eng2 = np.where(enr2 > 0, eval2 / enr2, 0.0)
    out["Engagement_Score"] = 0.5 * (eng1 + eng2)

    # 4. Academic Stress Index
    out["Academic_Stress_Index"] = (enr1 + enr2) - (app1 + app2)

    # 5. Financial Risk Indicator
    debtor = _col("Debtor", 0.0)
    tuition = _col("Tuition fees up to date", 1.0)
    scholarship = _col("Scholarship holder", 0.0)
    out["Financial_Risk"] = debtor + (1.0 - tuition) + (1.0 - scholarship)

    return out


def validate_batch_csv(
    df: pd.DataFrame,
    training_columns: List[str],
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate an uploaded batch CSV before inference.
    Returns (is_valid, missing_required_cols, missing_optional_cols).
    """
    if df is None or df.empty or len(df) == 0:
        return False, ["Uploaded CSV is empty (0 rows)."], []

    norm_cols = {c.replace("\ufeff", "").replace("\t", "").strip().replace(" ", "_") for c in df.columns}
    missing_req = [
        c for c in REQUIRED_RAW_COLUMNS
        if c.replace(" ", "_") not in norm_cols
    ]
    if missing_req:
        return False, missing_req, []

    engineered_names = {s["name"] for s in ENGINEERED_FEATURE_SPECS}
    missing_opt = [
        c for c in training_columns
        if c not in engineered_names and c not in norm_cols
    ]
    return True, [], missing_opt


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Engineer features, encode binary target (Dropout=1), and clean column names."""
    df_eng = engineer_features(df)
    if "Target" not in df_eng.columns:
        raise ValueError("Dataset must contain a 'Target' column.")

    y_raw = df_eng["Target"]
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.astype(int)
    else:
        y = y_raw.astype(str).str.strip().map(BINARY_TARGET_MAP)
        if y.isna().any():
            raise ValueError("Unrecognized Target labels; expected Dropout, Enrolled, or Graduate.")
        y = y.astype(int)

    X = df_eng.drop(columns=["Target"])
    if "id" in X.columns:
        X = X.drop(columns=["id"])

    # Coerce numeric; encode any unexpected object columns deterministically
    for col in X.columns:
        if X[col].dtype == "object" or "string" in str(X[col].dtype).lower():
            X[col] = pd.Categorical(X[col].astype(str)).codes
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X = clean_column_names(X)
    return X, y


def preprocess_inference_df(
    df: pd.DataFrame,
    training_columns: List[str],
    train_medians: Dict[str, float],
) -> pd.DataFrame:
    """Align and impute inference/batch data using training-set medians only."""
    df_eng = engineer_features(df)
    for drop_col in ("Target", "id"):
        if drop_col in df_eng.columns:
            df_eng = df_eng.drop(columns=[drop_col])

    for col in df_eng.columns:
        if df_eng[col].dtype == "object" or "string" in str(df_eng[col].dtype).lower():
            df_eng[col] = pd.Categorical(df_eng[col].astype(str)).codes
        else:
            df_eng[col] = pd.to_numeric(df_eng[col], errors="coerce")

    df_eng = clean_column_names(df_eng)

    for col in training_columns:
        if col not in df_eng.columns:
            df_eng[col] = train_medians.get(col, 0.0)
        else:
            df_eng[col] = df_eng[col].fillna(train_medians.get(col, 0.0))

    return df_eng[training_columns]


def train_and_evaluate(
    df: pd.DataFrame,
    use_smote: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train Logistic Regression, Random Forest, XGBoost, and LightGBM on a
    leakage-free stratified split (SMOTE applied ONLY to X_train).
    """
    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit medians ONLY on X_train
    train_medians = X_train.median(numeric_only=True).to_dict()
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    # Apply SMOTE ONLY on training data after split
    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train.copy(), y_train.copy()

    # Fit scaler ONLY on resampled training data
    scaler = StandardScaler()
    X_train_res_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=random_state, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    fitted_models: Dict[str, Any] = {}
    metrics_rows: List[Dict[str, Any]] = []
    confusion_matrices: Dict[str, np.ndarray] = {}
    roc_curves: Dict[str, Dict[str, Any]] = {}
    pr_curves: Dict[str, Dict[str, Any]] = {}
    test_probabilities: Dict[str, np.ndarray] = {}

    for name, clf in models.items():
        if name in SCALED_MODELS:
            clf.fit(X_train_res_scaled, y_train_res)
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            clf.fit(X_train_res, y_train_res)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

        fitted_models[name] = clf
        test_probabilities[name] = y_prob

        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        roc_curves[name] = {"fpr": fpr, "tpr": tpr, "auc": float(roc_auc)}

        prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        pr_curves[name] = {"precision": prec_c, "recall": rec_c, "pr_auc": float(pr_auc)}

        metrics_rows.append({
            "Model": name,
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
            "F1-Score": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
            "ROC-AUC": float(roc_auc),
            "PR-AUC": float(pr_auc),
        })

    comparison_df = pd.DataFrame(metrics_rows).set_index("Model")

    bundle = {
        "fitted_models": fitted_models,
        "comparison_df": comparison_df,
        "confusion_matrices": confusion_matrices,
        "roc_curves": roc_curves,
        "pr_curves": pr_curves,
        "test_probabilities": test_probabilities,
        "scaler": scaler,
        "train_medians": train_medians,
        "feature_columns": list(X_train.columns),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_res": X_train_res,
        "y_train_res": y_train_res,
        "smote_applied": use_smote,
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(bundle, save_path)

    return bundle


def load_or_train_pipeline(
    df: pd.DataFrame,
    artifact_path: Optional[str] = None,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """Load serialized joblib pipeline if present and compatible, else train & save."""
    if artifact_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        artifact_path = os.path.join(base_dir, "models", "eduguard_artifacts.joblib")

    if not force_retrain and os.path.exists(artifact_path):
        try:
            bundle = joblib.load(artifact_path)
            if set(CORE_MODEL_NAMES).issubset(bundle.get("fitted_models", {}).keys()):
                return bundle
        except Exception:
            pass

    return train_and_evaluate(df, use_smote=True, save_path=artifact_path)
