"""
Standalone End-to-End Verification Script for EduGuard (Tasks 3, 4, and PDF/KB checks).

Run in an approved Python 3.11 or 3.12 environment where native .pyd extensions
(numpy, pandas, scikit-learn, xgboost, lightgbm, shap) are permitted:
    python verify_all.py
"""

from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from early_warning import (
    build_cohort_summary_pdf,
    build_student_pdf_dossier,
    classify_risk_level,
    generate_intervention_plan,
)
from knowledge_base import KNOWLEDGE_CATEGORIES, PRESET_QUESTIONS, ResearchKnowledgeBase
from ml_pipeline import (
    CORE_MODEL_NAMES,
    inspect_dataset_quality,
    load_raw_dataset,
    preprocess_inference_df,
    train_and_evaluate,
    validate_batch_csv,
)
from xai_engine import (
    build_lime_explainer,
    compute_global_shap,
    compute_local_shap,
    explain_instance_lime,
    generate_grounded_nl_explanation,
)


def main() -> None:
    print("=== 1. Knowledge Base & Early Warning Checks ===")
    kb = ResearchKnowledgeBase()
    assert len(kb.papers) == 30, f"Expected 30 papers, got {len(kb.papers)}"
    assert len(KNOWLEDGE_CATEGORIES) == 14
    for q in PRESET_QUESTIONS:
        assert len(kb.query(q, top_k=3)) > 0
    print("KB OK: 30 papers, 14 categories, 7 preset queries passed.")

    print("\n=== 2. Dataset & ML Pipeline Execution (Task 3) ===")
    df = load_raw_dataset()
    qual = inspect_dataset_quality(df)
    print(f"Dataset shape: {qual['rows']} x {qual['columns']} | Missing: {qual['total_missing']} | Duplicates: {qual['duplicate_rows']}")
    print(f"Binary target counts: {qual['binary_target_counts']}")

    artifact_path = os.path.join(BASE_DIR, "models", "eduguard_artifacts.joblib")
    bundle = train_and_evaluate(df, use_smote=True, save_path=artifact_path)
    assert set(bundle["y_train"].unique()) == {0, 1}
    assert len(bundle["X_train_res"]) > len(bundle["X_train"])  # SMOTE applied to train only
    assert len(bundle["X_test"]) == len(bundle["y_test"])

    print("\nMeasured Model Evaluation Results (Test Split):")
    print(bundle["comparison_df"].to_string())
    for m in CORE_MODEL_NAMES:
        print(f"\nConfusion Matrix ({m}):\n{bundle['confusion_matrices'][m]}")

    print("\n=== 3. SHAP & LIME Verification (Task 4) ===")
    X_train = bundle["X_train"]
    X_test = bundle["X_test"]
    scaler = bundle["scaler"]
    inst = X_test.iloc[[0]]

    for m in CORE_MODEL_NAMES:
        model = bundle["fitted_models"][m]
        shap_global, _ = compute_global_shap(model, X_train, m, scaler, sample_size=50)
        assert shap_global.shape == (min(50, len(X_train)), X_train.shape[1])
        local_df, _ = compute_local_shap(model, X_train, inst, m, scaler)
        nl = generate_grounded_nl_explanation(local_df, top_n=3)
        assert len(nl) > 0
        print(f"SHAP OK for {m}: global={shap_global.shape}, top_driver={local_df.iloc[0]['Feature']}")

    lime_exp = build_lime_explainer(X_train)
    df_lime, lime_html = explain_instance_lime(lime_exp, bundle["fitted_models"]["Random Forest"], inst, "Random Forest", scaler)
    assert len(df_lime) > 0 and len(lime_html) > 100
    print(f"LIME OK: {len(df_lime)} local rules and raw HTML generated.")

    print("\n=== 4. Batch CSV Validation Verification (Phase 7) ===")
    import pandas as pd
    f_cols = bundle["feature_columns"]
    # 1. Valid CSV
    ok1, _, _ = validate_batch_csv(df.head(5).drop(columns=["Target"]), f_cols)
    assert ok1
    # 2. Missing optional column
    ok2, _, opt2 = validate_batch_csv(df.head(5).drop(columns=["Target", "GDP"]), f_cols)
    assert ok2 and "GDP" in opt2
    # 3. Missing required column
    ok3, req3, _ = validate_batch_csv(df.head(5).drop(columns=["Target", "Debtor"]), f_cols)
    assert not ok3 and "Debtor" in req3
    # 4. Additional column
    df_extra = df.head(5).drop(columns=["Target"]).copy()
    df_extra["Extra_Col"] = 123
    ok4, _, _ = validate_batch_csv(df_extra, f_cols)
    assert ok4 and preprocess_inference_df(df_extra, f_cols, bundle["train_medians"]).shape[1] == len(f_cols)
    # 5. Empty CSV
    ok5, req5, _ = validate_batch_csv(pd.DataFrame(), f_cols)
    assert not ok5
    print("Batch CSV validation OK for all 5 edge cases.")

    print("\n=== 5. PDF Export Verification ===")
    prob = float(bundle["test_probabilities"]["XGBoost"][0])
    lvl = classify_risk_level(prob)
    plan = generate_intervention_plan(inst.iloc[0].to_dict(), prob)
    pdf_s = build_student_pdf_dossier("0", inst.iloc[0].to_dict(), prob, lvl, plan, nl)
    pdf_c = build_cohort_summary_pdf("XGBoost", bundle["comparison_df"].loc["XGBoost"].to_dict(), {"Critical": 1, "High": 1, "Medium": 1, "Low": 1}, len(X_test))
    assert len(pdf_s) > 500 and len(pdf_c) > 500
    print(f"PDF Export OK: student={len(pdf_s)} bytes, cohort={len(pdf_c)} bytes.")
    print("\nALL VERIFICATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
