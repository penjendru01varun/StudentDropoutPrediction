"""
EduGuard Explainable AI (XAI) Engine (Phase 6).

Provides SHAP (global importance, beeswarm, local waterfall/contributions) and
LIME (local tabular rules) for the binary Dropout (1) vs Non-Dropout (0) models,
along with grounded natural-language explanations derived strictly from actual
SHAP/LIME outputs and observed student feature values.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ml_pipeline import SCALED_MODELS


def extract_dropout_shap_values(shap_values: Any) -> np.ndarray:
    """Extract SHAP values corresponding to the positive Dropout class (1)."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        idx = 1 if arr.shape[2] > 1 else 0
        return arr[:, :, idx]
    return arr


def build_shap_explainer(
    model: Any,
    X_train: pd.DataFrame,
    model_name: str,
    scaler: Optional[Any] = None,
) -> Tuple[Any, Optional[np.ndarray]]:
    """Initialize an appropriate SHAP explainer for the selected model."""
    bg = X_train.sample(n=min(100, len(X_train)), random_state=42)
    if model_name in SCALED_MODELS and scaler is not None:
        bg_scaled = scaler.transform(bg)
        return shap.LinearExplainer(model, bg_scaled), bg_scaled
    return shap.TreeExplainer(model), None


def compute_global_shap(
    model: Any,
    X_train: pd.DataFrame,
    model_name: str,
    scaler: Optional[Any] = None,
    sample_size: int = 120,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Compute global SHAP values for the Dropout (1) class on a background sample."""
    X_sample = X_train.sample(n=min(sample_size, len(X_train)), random_state=42)
    explainer, _ = build_shap_explainer(model, X_train, model_name, scaler)
    if model_name in SCALED_MODELS and scaler is not None:
        raw_vals = explainer.shap_values(scaler.transform(X_sample))
    else:
        raw_vals = explainer.shap_values(X_sample)
    shap_c1 = extract_dropout_shap_values(raw_vals)
    return shap_c1, X_sample


def compute_local_shap(
    model: Any,
    X_train: pd.DataFrame,
    instance_df: pd.DataFrame,
    model_name: str,
    scaler: Optional[Any] = None,
) -> Tuple[pd.DataFrame, float]:
    """Compute local SHAP contributions and base value for a single student."""
    explainer, _ = build_shap_explainer(model, X_train, model_name, scaler)
    if model_name in SCALED_MODELS and scaler is not None:
        X_in = scaler.transform(instance_df)
    else:
        X_in = instance_df

    raw_vals = explainer.shap_values(X_in)
    contrib = extract_dropout_shap_values(raw_vals).flatten()

    base_val = 0.0
    ev = getattr(explainer, "expected_value", 0.0)
    if isinstance(ev, (list, np.ndarray)):
        ev_arr = np.asarray(ev).flatten()
        base_val = float(ev_arr[1] if len(ev_arr) > 1 else ev_arr[0])
    else:
        base_val = float(ev)

    df_local = pd.DataFrame({
        "Feature": list(instance_df.columns),
        "Value": instance_df.iloc[0].values,
        "SHAP_Contribution": contrib,
        "Abs_Contribution": np.abs(contrib),
    }).sort_values("Abs_Contribution", ascending=False)

    return df_local, base_val


def build_lime_explainer(X_train: pd.DataFrame) -> lime.lime_tabular.LimeTabularExplainer:
    """Create a LIME tabular explainer aligned with binary Non-Dropout / Dropout."""
    return lime.lime_tabular.LimeTabularExplainer(
        training_data=np.asarray(X_train),
        feature_names=list(X_train.columns),
        class_names=["Non-Dropout", "Dropout"],
        mode="classification",
        random_state=42,
    )


def explain_instance_lime(
    lime_explainer: lime.lime_tabular.LimeTabularExplainer,
    model: Any,
    instance_df: pd.DataFrame,
    model_name: str,
    scaler: Optional[Any] = None,
    num_features: int = 10,
) -> Tuple[pd.DataFrame, str]:
    """Generate LIME local rules (DataFrame) and raw HTML for Dropout (1)."""
    if model_name in SCALED_MODELS and scaler is not None:
        predict_fn = lambda x: model.predict_proba(scaler.transform(x))
    else:
        predict_fn = model.predict_proba

    exp = lime_explainer.explain_instance(
        data_row=instance_df.values[0],
        predict_fn=predict_fn,
        num_features=num_features,
        labels=(1,),
    )
    rules = exp.as_list(label=1)
    df_lime = pd.DataFrame(rules, columns=["Rule", "Weight"])
    df_lime["Direction"] = df_lime["Weight"].apply(
        lambda w: "Increases Dropout Risk" if w > 0 else "Protective (Reduces Risk)"
    )
    return df_lime, exp.as_html()


def generate_grounded_nl_explanation(
    local_shap_df: pd.DataFrame,
    top_n: int = 5,
) -> List[str]:
    """
    Generate natural-language explanation bullets strictly from actual local SHAP
    contributions and observed feature values. Distinguishes model associations
    from proven causal relationships.
    """
    bullets: List[str] = []
    top_rows = local_shap_df.head(top_n)
    for _, r in top_rows.iterrows():
        feat = str(r["Feature"]).replace("_", " ")
        val = float(r["Value"])
        contrib = float(r["SHAP_Contribution"])
        if contrib > 0:
            bullets.append(
                f"**{feat}** (observed value: `{val:.2f}`) is associated with **higher** "
                f"predicted dropout risk (SHAP contribution: `+{contrib:.4f}`)."
            )
        elif contrib < 0:
            bullets.append(
                f"**{feat}** (observed value: `{val:.2f}`) is associated with **lower** "
                f"predicted dropout risk (SHAP contribution: `{contrib:.4f}`)."
            )
    if not bullets:
        bullets.append("No non-zero feature contributions were recorded for this prediction.")
    return bullets


def create_shap_waterfall_figure(
    local_shap_df: pd.DataFrame,
    base_val: float,
    instance_df: pd.DataFrame,
    max_display: int = 10,
):
    """Build a matplotlib SHAP waterfall figure for an individual student."""
    # Reorder contributions to match instance_df columns
    contrib_map = dict(zip(local_shap_df["Feature"], local_shap_df["SHAP_Contribution"]))
    ordered_vals = np.array([contrib_map[c] for c in instance_df.columns], dtype=float)
    exp = shap.Explanation(
        values=ordered_vals,
        base_values=base_val,
        data=instance_df.iloc[0].values,
        feature_names=list(instance_df.columns),
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    fig.patch.set_facecolor("#161333")
    for a in fig.axes:
        a.set_facecolor("#161333")
        a.tick_params(colors="#e2e8f0", labelsize=8)
        for txt in a.texts:
            txt.set_color("#e2e8f0")
    plt.tight_layout()
    return fig
