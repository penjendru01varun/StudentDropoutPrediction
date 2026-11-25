"""
EduGuard — Student Dropout Prediction & Early Warning System (Streamlit App).

Integrates:
- Leakage-free binary ML pipeline (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Dual Explainable AI (SHAP global/local/waterfall + LIME local rules)
- Configurable-threshold Early Warning & Decision-Support Intervention workflow
- Searchable 30-Paper Research Knowledge Base (BM25 local RAG)
- Premium Dark Glassmorphism Analytics UI
"""

import io
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
import streamlit.components.v1 as components

from early_warning import (
    DEFAULT_RISK_THRESHOLDS,
    build_cohort_summary_pdf,
    build_student_pdf_dossier,
    classify_risk_level,
    generate_intervention_plan,
)
from knowledge_base import KNOWLEDGE_CATEGORIES, PRESET_QUESTIONS, ResearchKnowledgeBase
from ml_pipeline import (
    BINARY_TARGET_MAP,
    CORE_MODEL_NAMES,
    ENGINEERED_FEATURE_SPECS,
    SCALED_MODELS,
    inspect_dataset_quality,
    load_or_train_pipeline,
    load_raw_dataset,
    preprocess_inference_df,
    validate_batch_csv,
)
from xai_engine import (
    build_lime_explainer,
    compute_global_shap,
    compute_local_shap,
    create_shap_waterfall_figure,
    explain_instance_lime,
    generate_grounded_nl_explanation,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="EduGuard | Student Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============= PREMIUM DARK GLASSMORPHISM CSS =============
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 52%, #24243e 100%);
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.78);
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 2.15rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #f093fb 55%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }

    .subtitle {
        color: rgba(226, 232, 240, 0.76);
        font-size: 0.96rem;
        margin-bottom: 1.4rem;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.045);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 16px;
        padding: 1.2rem 1.35rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .glass-card:hover {
        border-color: rgba(102, 126, 234, 0.45);
    }

    .kpi-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: rgba(226, 232, 240, 0.68);
        margin: 0;
    }

    .kpi-value {
        font-family: 'Poppins', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        margin: 0.2rem 0 0 0;
        color: #ffffff;
    }

    .risk-badge {
        display: inline-block;
        padding: 0.22rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .risk-critical { background: rgba(245, 87, 108, 0.22); color: #fda4af; border: 1px solid #f5576c; }
    .risk-high     { background: rgba(239, 68, 68, 0.20); color: #fca5a5; border: 1px solid #ef4444; }
    .risk-medium   { background: rgba(245, 158, 11, 0.20); color: #fcd34d; border: 1px solid #f59e0b; }
    .risk-low      { background: rgba(16, 185, 129, 0.20); color: #6ee7b7; border: 1px solid #10b981; }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: #ffffff;
        font-weight: 600;
        border-radius: 10px;
        transition: opacity 0.2s ease, transform 0.15s ease;
    }
    .stButton > button[kind="primary"]:hover {
        opacity: 0.92;
        transform: translateY(-1px);
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def get_dataset():
    return load_raw_dataset()


@st.cache_resource
def get_ml_bundle(force_retrain: bool = False):
    df = get_dataset()
    if df.empty:
        return None
    return load_or_train_pipeline(df, force_retrain=force_retrain)


@st.cache_resource
def get_kb():
    return ResearchKnowledgeBase()


def _plotly_dark(fig, height: int = 320):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f1f5f9",
        height=height,
        margin=dict(t=32, b=18, l=16, r=16),
    )
    return fig


def _predict_probs(model, X_df: pd.DataFrame, model_name: str, scaler) -> np.ndarray:
    if model_name in SCALED_MODELS:
        return model.predict_proba(scaler.transform(X_df))[:, 1]
    return model.predict_proba(X_df)[:, 1]


def main():
    df_raw = get_dataset()
    if df_raw.empty:
        st.error("Dataset not found. Place `student_dropout_data.csv` in `data/` or `dataset.csv` in root.")
        return

    bundle = get_ml_bundle()
    kb = get_kb()

    if "navigation" not in st.session_state:
        st.session_state.navigation = "🏠 Dashboard"
    if "selected_model_name" not in st.session_state:
        st.session_state.selected_model_name = "XGBoost"
    if "selected_student_id" not in st.session_state:
        st.session_state.selected_student_id = None
    if "recent_predictions" not in st.session_state:
        st.session_state.recent_predictions = []

    # ============= SIDEBAR =============
    st.sidebar.markdown(
        """
        <div style="text-align:center; padding: 0.4rem 0 0.8rem 0;">
            <h2 style="margin:0; font-family:'Poppins',sans-serif; font-weight:800;
                       background:linear-gradient(135deg,#667eea,#f093fb);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                🎓 EduGuard
            </h2>
            <p style="margin:2px 0 0 0; font-size:0.76rem; color:rgba(226,232,240,0.65); letter-spacing:0.08em;">
                EARLY WARNING & XAI PLATFORM
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    menu = [
        "🏠 Dashboard",
        "📊 Data Explorer",
        "🤖 Model Hub",
        "⚠️ Early Warning System",
        "🔮 Student Predictor",
        "📈 Reports",
        "📚 Research Knowledge Base",
    ]
    idx = menu.index(st.session_state.navigation) if st.session_state.navigation in menu else 0
    choice = st.sidebar.radio("Navigation", menu, index=idx, label_visibility="collapsed")
    if choice != st.session_state.navigation:
        st.session_state.navigation = choice
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Active Classifier")
    st.session_state.selected_model_name = st.sidebar.selectbox(
        "Model",
        CORE_MODEL_NAMES,
        index=CORE_MODEL_NAMES.index(st.session_state.selected_model_name),
        label_visibility="collapsed",
    )

    if st.sidebar.button("🔄 Retrain & Save Models", use_container_width=True):
        st.cache_resource.clear()
        bundle = get_ml_bundle(force_retrain=True)
        st.sidebar.success("Models retrained and saved via joblib.")

    st.sidebar.markdown("### 🎚️ Prototype Risk Thresholds")
    t_crit = st.sidebar.slider("Critical (>=)", 0.75, 0.99, DEFAULT_RISK_THRESHOLDS["critical"], 0.01)
    t_high = st.sidebar.slider("High (>=)", 0.50, t_crit - 0.05, min(DEFAULT_RISK_THRESHOLDS["high"], t_crit - 0.05), 0.01)
    t_med = st.sidebar.slider("Medium (>=)", 0.20, t_high - 0.05, min(DEFAULT_RISK_THRESHOLDS["medium"], t_high - 0.05), 0.01)
    thresholds = {"critical": t_crit, "high": t_high, "medium": t_med}
    st.sidebar.caption("Prototype thresholds for educator decision support.")

    # Active model & test-split predictions
    active_name = st.session_state.selected_model_name
    active_model = bundle["fitted_models"][active_name]
    scaler = bundle["scaler"]
    X_train = bundle["X_train"]
    X_test = bundle["X_test"]
    y_test = bundle["y_test"]
    comparison_df = bundle["comparison_df"]
    test_probs = bundle["test_probabilities"][active_name]

    risk_levels = pd.Series(test_probs).apply(lambda p: classify_risk_level(p, thresholds))
    critical_count = int((risk_levels == "Critical").sum())
    high_count = int((risk_levels == "High").sum())
    at_risk_count = int(risk_levels.isin(["Critical", "High"]).sum())

    # =========================================================================
    # 1. MAIN DASHBOARD
    # =========================================================================
    if st.session_state.navigation == "🏠 Dashboard":
        st.markdown('<h1 class="main-title">🎓 EduGuard Early Warning Overview</h1>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="subtitle">Binary Dropout vs. Non-Dropout decision-support dashboard | Active Model: <b>{active_name}</b> | Test Split: {len(X_test):,} students (from {len(df_raw):,} cohort records)</p>',
            unsafe_allow_html=True,
        )

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f'<div class="glass-card"><p class="kpi-label">Total Students Analyzed</p><p class="kpi-value">{len(X_test):,}</p></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="glass-card"><p class="kpi-label">Dropout Risk (High+Crit)</p><p class="kpi-value" style="color:#f5576c;">{at_risk_count:,}</p></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="glass-card"><p class="kpi-label">Critical Risk (≥{t_crit:.0%})</p><p class="kpi-value" style="color:#f093fb;">{critical_count:,}</p></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="glass-card"><p class="kpi-label">High Risk ({t_high:.0%}–{t_crit:.0%})</p><p class="kpi-value" style="color:#fca5a5;">{high_count:,}</p></div>', unsafe_allow_html=True)

        c_left, c_right = st.columns([1.15, 1.15])
        with c_left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Risk Distribution (Test Cohort)")
            rc = risk_levels.value_counts().reindex(["Critical", "High", "Medium", "Low"], fill_value=0).reset_index()
            rc.columns = ["Risk Level", "Students"]
            fig_pie = px.pie(
                rc, values="Students", names="Risk Level", hole=0.48, color="Risk Level",
                color_discrete_map={"Critical": "#f093fb", "High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
            )
            st.plotly_chart(_plotly_dark(fig_pie, 290), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c_right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 📈 Predicted Dropout Probability Distribution")
            fig_hist = px.histogram(
                x=test_probs, nbins=25, labels={"x": "Predicted Dropout Probability", "y": "Students"},
                color_discrete_sequence=["#667eea"],
            )
            st.plotly_chart(_plotly_dark(fig_hist, 290), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        c_imp, c_perf = st.columns([1.2, 1.0])
        with c_imp:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🔍 Top Global Risk Drivers (SHAP)")
            try:
                shap_c1, X_sample = compute_global_shap(active_model, X_train, active_name, scaler, sample_size=100)
                mean_abs = np.abs(shap_c1).mean(axis=0)
                imp_df = pd.DataFrame({"Feature": X_sample.columns, "Mean |SHAP|": mean_abs}).sort_values("Mean |SHAP|", ascending=False).head(10)
                fig_imp = px.bar(imp_df, x="Mean |SHAP|", y="Feature", orientation="h", color="Mean |SHAP|", color_continuous_scale="Purples")
                fig_imp.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
                st.plotly_chart(_plotly_dark(fig_imp, 310), use_container_width=True)
            except Exception as e:
                st.info(f"SHAP global importance unavailable: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        with c_perf:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🤖 Model Performance Overview (Test Split)")
            st.dataframe(comparison_df.style.format("{:.4f}"), use_container_width=True)
            if st.session_state.recent_predictions:
                st.markdown("**Recent Predictor Activity**")
                st.dataframe(pd.DataFrame(st.session_state.recent_predictions[-5:]), use_container_width=True)
            else:
                st.caption("Run predictions in Student Predictor to log recent session activity here.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🚨 Priority At-Risk Students (Test Split)")
        triage_df = X_test.copy()
        triage_df["Student_ID"] = triage_df.index.astype(str)
        triage_df["Dropout_Probability"] = test_probs
        triage_df["Risk_Level"] = risk_levels.values
        top_risk = triage_df[triage_df["Risk_Level"].isin(["Critical", "High"])].sort_values("Dropout_Probability", ascending=False).head(5)
        if not top_risk.empty:
            for _, trow in top_risk.iterrows():
                tc1, tc2 = st.columns([5, 1])
                with tc1:
                    st.markdown(
                        f"**Student #{trow['Student_ID']}** | Risk: `{trow['Risk_Level']}` (`{trow['Dropout_Probability']:.1%}`) | "
                        f"Stress Index: `{trow.get('Academic_Stress_Index', 0.0):.1f}` | Financial Risk: `{trow.get('Financial_Risk', 0.0):.1f}`"
                    )
                with tc2:
                    if st.button("🔍 Intervene", key=f"dash_int_{trow['Student_ID']}"):
                        st.session_state.selected_student_id = trow["Student_ID"]
                        st.session_state.navigation = "⚠️ Early Warning System"
                        st.rerun()
        else:
            st.info("No students exceed the current High/Critical prototype thresholds.")
        st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # 2. DATA EXPLORER
    # =========================================================================
    elif st.session_state.navigation == "📊 Data Explorer":
        st.markdown('<h1 class="main-title">📊 Data Explorer & Quality Inspection</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">UCI Student Dropout & Academic Success dataset (4,424 records, 36 original features + Target).</p>', unsafe_allow_html=True)

        qual = inspect_dataset_quality(df_raw)
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Records", f"{qual['rows']:,}")
        q2.metric("Original Features", f"{qual['features']}")
        q3.metric("Missing Cells", f"{qual['total_missing']}")
        q4.metric("Duplicate Rows", f"{qual['duplicate_rows']}")

        t1, t2, t3, t4 = st.tabs(["📋 Preview & Schema", "📊 Class & Feature Distributions", "🔗 Correlations", "⚙️ Engineered Features"])

        with t1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Dataset Preview (First 20 Rows)")
            st.dataframe(df_raw.head(20), use_container_width=True)
            st.subheader("Statistical Summary")
            st.dataframe(df_raw.describe().T, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with t2:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                orig_df = pd.DataFrame(list(qual["original_target_counts"].items()), columns=["Class", "Count"])
                fig_o = px.pie(orig_df, values="Count", names="Class", title="Original 3-Class Distribution", hole=0.4)
                st.plotly_chart(_plotly_dark(fig_o, 300), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                bin_df = pd.DataFrame(list(qual["binary_target_counts"].items()), columns=["Class", "Count"])
                fig_b = px.bar(bin_df, x="Class", y="Count", color="Class", title="Binary Target (Dropout=1 vs Non-Dropout=0)",
                               color_discrete_map={"Dropout (1)": "#f5576c", "Non-Dropout (0)": "#10b981"})
                st.plotly_chart(_plotly_dark(fig_b, 300), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            num_cols = [c for c in df_raw.columns if c != "Target" and pd.api.types.is_numeric_dtype(df_raw[c])]
            sel_feat = st.selectbox("Inspect Feature Distribution by Target:", num_cols, index=min(19, len(num_cols) - 1))
            fig_fd = px.box(df_raw, x="Target", y=sel_feat, color="Target", title=f"{sel_feat} by Outcome")
            st.plotly_chart(_plotly_dark(fig_fd, 330), use_container_width=True)

        with t3:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            df_corr = df_raw.copy()
            df_corr["Binary_Dropout"] = df_corr["Target"].astype(str).map(BINARY_TARGET_MAP)
            corr_series = df_corr.select_dtypes(include=[np.number]).corr()["Binary_Dropout"].drop("Binary_Dropout").sort_values(key=abs, ascending=False).head(15)
            fig_corr = px.bar(x=corr_series.values, y=corr_series.index, orientation="h",
                              labels={"x": "Pearson Correlation with Dropout (1)", "y": "Feature"},
                              color=corr_series.values, color_continuous_scale="RdBu_r")
            fig_corr.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(_plotly_dark(fig_corr, 420), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with t4:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Verified Engineered Features (Row-Wise, Zero Leakage)")
            st.dataframe(pd.DataFrame(ENGINEERED_FEATURE_SPECS), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # 3. MODEL HUB
    # =========================================================================
    elif st.session_state.navigation == "🤖 Model Hub":
        st.markdown('<h1 class="main-title">🤖 Machine Learning Model Hub</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Fair comparison of Logistic Regression, Random Forest, XGBoost, and LightGBM on the identical stratified test split (SMOTE applied to training data only).</p>', unsafe_allow_html=True)

        cols = st.columns(4)
        for i, m_name in enumerate(CORE_MODEL_NAMES):
            row = comparison_df.loc[m_name]
            with cols[i]:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <h4 style="margin:0 0 8px 0; color:#f093fb;">{m_name}</h4>
                        <p style="margin:2px 0; font-size:0.85rem;">Accuracy: <b>{row['Accuracy']:.3f}</b></p>
                        <p style="margin:2px 0; font-size:0.85rem;">Precision: <b>{row['Precision']:.3f}</b></p>
                        <p style="margin:2px 0; font-size:0.85rem;">Recall (Dropout): <b>{row['Recall']:.3f}</b></p>
                        <p style="margin:2px 0; font-size:0.85rem;">F1-Score: <b>{row['F1-Score']:.3f}</b></p>
                        <p style="margin:2px 0; font-size:0.85rem;">ROC-AUC: <b>{row['ROC-AUC']:.3f}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Measured Benchmark Table (Test Split)")
        st.dataframe(comparison_df.style.highlight_max(axis=0, color="rgba(102, 126, 234, 0.35)").format("{:.4f}"), use_container_width=True)
        st.caption("Models and preprocessing artifacts are serialized to `models/eduguard_artifacts.joblib`.")
        st.markdown("</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig_roc = go.Figure()
            for m_name, roc_data in bundle["roc_curves"].items():
                fig_roc.add_trace(go.Scatter(x=roc_data["fpr"], y=roc_data["tpr"], mode="lines", name=f"{m_name} (AUC={roc_data['auc']:.3f})"))
            fig_roc.add_shape(type="line", line=dict(dash="dash", color="rgba(255,255,255,0.4)"), x0=0, x1=1, y0=0, y1=1)
            fig_roc.update_layout(title="ROC Curves (All 4 Models)", xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(_plotly_dark(fig_roc, 320), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            cm = bundle["confusion_matrices"][active_name]
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=["Pred Non-Dropout (0)", "Pred Dropout (1)"], y=["True Non-Dropout (0)", "True Dropout (1)"],
                colorscale="Purples", text=cm, texttemplate="%{text}",
            ))
            fig_cm.update_layout(title=f"Confusion Matrix — {active_name}")
            st.plotly_chart(_plotly_dark(fig_cm, 320), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader(f"Global SHAP Summary (Beeswarm) — {active_name}")
        try:
            shap_c1, X_sample = compute_global_shap(active_model, X_train, active_name, scaler, sample_size=120)
            fig_bs, ax = plt.subplots(figsize=(8, 5))
            shap.summary_plot(shap_c1, X_sample, show=False)
            fig_bs.patch.set_facecolor("#161333")
            ax.set_facecolor("#161333")
            ax.tick_params(colors="#e2e8f0", labelsize=8)
            for item in [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
                item.set_color("#e2e8f0")
            st.pyplot(fig_bs, clear_figure=True)
        except Exception as e:
            st.warning(f"SHAP beeswarm plot unavailable: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # 4. EARLY WARNING SYSTEM
    # =========================================================================
    elif st.session_state.navigation == "⚠️ Early Warning System":
        st.markdown('<h1 class="main-title">⚠️ Early Warning & Intervention Triage</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Prioritized decision-support directory using configurable prototype thresholds and SHAP-backed intervention plans.</p>', unsafe_allow_html=True)

        ews_tab1, ews_tab2 = st.tabs(["📋 Student Risk Directory", "📤 Batch CSV Prediction"])

        with ews_tab1:
            test_df = X_test.copy()
            test_df["Student_ID"] = test_df.index.astype(str)
            test_df["Dropout_Probability"] = test_probs
            test_df["Risk_Level"] = risk_levels.values

            f1, f2, f3 = st.columns([1.4, 1.0, 1.2])
            with f1:
                sel_risks = st.multiselect("Filter Risk Levels:", ["Critical", "High", "Medium", "Low"], default=["Critical", "High"])
            with f2:
                search_id = st.text_input("Search Student ID:", value=str(st.session_state.selected_student_id or ""))
            with f3:
                sort_desc = st.checkbox("Sort by Highest Risk", value=True)

            filt = test_df[test_df["Risk_Level"].isin(sel_risks)] if sel_risks else test_df.copy()
            if search_id.strip():
                filt = filt[filt["Student_ID"].str.contains(search_id.strip())]
            if sort_desc:
                filt = filt.sort_values("Dropout_Probability", ascending=False)

            show_cols = ["Student_ID", "Risk_Level", "Dropout_Probability", "Credit_Completion_Rate_1st_Sem", "Grade_Drop", "Academic_Stress_Index", "Financial_Risk"]
            avail_cols = [c for c in show_cols if c in filt.columns]
            st.dataframe(filt[avail_cols].head(100), use_container_width=True)

            if not filt.empty:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                sel_sid = st.selectbox("Inspect Student Dossier:", filt["Student_ID"].head(50).tolist())
                row_full = filt[filt["Student_ID"] == sel_sid].iloc[0]
                inst_df = pd.DataFrame([row_full[X_train.columns]])
                prob_s = float(row_full["Dropout_Probability"])
                lvl_s = str(row_full["Risk_Level"])

                local_df, _ = compute_local_shap(active_model, X_train, inst_df, active_name, scaler)
                nl_bullets = generate_grounded_nl_explanation(local_df, top_n=4)
                plan = generate_intervention_plan(row_full.to_dict(), prob_s, thresholds)

                c_a, c_b = st.columns(2)
                with c_a:
                    st.markdown(f"#### 🧠 SHAP Risk Drivers ({lvl_s} — {prob_s:.1%})")
                    for b in nl_bullets:
                        st.markdown(f"- {b}")
                with c_b:
                    st.markdown("#### 💡 Recommended Decision-Support Actions")
                    for sec, key in [("Academic", "academic"), ("Financial", "financial"), ("Engagement", "engagement")]:
                        if plan[key]:
                            st.markdown(f"**{sec}:**")
                            for item in plan[key]:
                                st.markdown(f"- {item}")
                    st.caption(plan["advisory_note"])

                pdf_bytes = build_student_pdf_dossier(sel_sid, row_full.to_dict(), prob_s, lvl_s, plan, nl_bullets)
                st.download_button("📥 Download Student PDF Dossier", data=pdf_bytes, file_name=f"eduguard_student_{sel_sid}.pdf", mime="application/pdf")
                st.markdown("</div>", unsafe_allow_html=True)

        with ews_tab2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            batch_file = st.file_uploader("Upload Student Cohort CSV for Batch Scoring", type=["csv"])
            if batch_file is not None:
                try:
                    raw_b = pd.read_csv(batch_file, sep=None, engine="python")
                    is_valid, missing_req, missing_opt = validate_batch_csv(raw_b, bundle["feature_columns"])
                    if not is_valid:
                        st.error(f"Batch CSV validation failed. Missing required columns or empty file: {', '.join(missing_req)}")
                    else:
                        if missing_opt:
                            st.warning(f"Optional columns missing ({len(missing_opt)}); imputed using training-set medians.")
                        X_b = preprocess_inference_df(raw_b, bundle["feature_columns"], bundle["train_medians"])
                        probs_b = _predict_probs(active_model, X_b, active_name, scaler)
                        out_b = raw_b.copy()
                        out_b["Dropout_Probability"] = np.round(probs_b, 4)
                        out_b["Risk_Level"] = [classify_risk_level(p, thresholds) for p in probs_b]
                        st.success(f"Scored {len(out_b):,} records using {active_name}.")
                        st.dataframe(out_b[["Dropout_Probability", "Risk_Level"] + list(out_b.columns[:5])].head(50), use_container_width=True)
                        st.download_button("📥 Download Scored Batch CSV", data=out_b.to_csv(index=False).encode("utf-8"), file_name="eduguard_batch_predictions.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Batch scoring error: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # 5. STUDENT PREDICTOR
    # =========================================================================
    elif st.session_state.navigation == "🔮 Student Predictor":
        st.markdown('<h1 class="main-title">🔮 Individual Student Risk Predictor</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Enter supported UCI student attributes to compute dropout probability, SHAP waterfall & LIME explanations, and decision-support recommendations.</p>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age at enrollment", 17, 65, 20)
            gender = st.selectbox("Gender", ["Female (0)", "Male (1)"])
            displaced = st.checkbox("Displaced", value=False)
            admission_grade = st.slider("Admission grade", 95.0, 190.0, 125.0)
        with c2:
            debtor = st.checkbox("Debtor (Outstanding balance)", value=False)
            tuition = st.checkbox("Tuition fees up to date", value=True)
            scholarship = st.checkbox("Scholarship holder", value=False)
            unemp = st.slider("Unemployment rate (%)", 7.0, 16.5, 11.0)
        with c3:
            enr1 = st.slider("1st sem enrolled units", 0, 15, 6)
            app1 = st.slider("1st sem approved units", 0, 15, 5)
            grd1 = st.slider("1st sem average grade", 0.0, 20.0, 12.5)
            enr2 = st.slider("2nd sem enrolled units", 0, 15, 6)
            app2 = st.slider("2nd sem approved units", 0, 15, 5)
            grd2 = st.slider("2nd sem average grade", 0.0, 20.0, 12.0)

        if st.button("🔮 Run Risk Prediction & XAI", type="primary"):
            base_row = {col: bundle["train_medians"].get(col, 0.0) for col in bundle["feature_columns"]}
            raw_inputs = {
                "Age at enrollment": age,
                "Gender": 1 if "Male" in gender else 0,
                "Displaced": int(displaced),
                "Admission grade": admission_grade,
                "Debtor": int(debtor),
                "Tuition fees up to date": int(tuition),
                "Scholarship holder": int(scholarship),
                "Unemployment rate": unemp,
                "Curricular units 1st sem (enrolled)": enr1,
                "Curricular units 1st sem (evaluations)": enr1,
                "Curricular units 1st sem (approved)": app1,
                "Curricular units 1st sem (grade)": grd1,
                "Curricular units 2nd sem (enrolled)": enr2,
                "Curricular units 2nd sem (evaluations)": enr2,
                "Curricular units 2nd sem (approved)": app2,
                "Curricular units 2nd sem (grade)": grd2,
            }
            X_in = preprocess_inference_df(pd.DataFrame([raw_inputs]), bundle["feature_columns"], bundle["train_medians"])
            # Preserve median defaults for unedited columns
            for c in bundle["feature_columns"]:
                if c not in X_in.columns:
                    X_in[c] = base_row[c]

            prob = float(_predict_probs(active_model, X_in, active_name, scaler)[0])
            lvl = classify_risk_level(prob, thresholds)

            st.session_state.recent_predictions.append({"Model": active_name, "Probability": f"{prob:.1%}", "Risk_Level": lvl})

            g_col, r_col = st.columns([1.0, 1.2])
            with g_col:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob * 100,
                    title={"text": "Dropout Probability (%)", "font": {"color": "#f8fafc"}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#764ba2"},
                        "steps": [
                            {"range": [0, t_med * 100], "color": "rgba(16,185,129,0.22)"},
                            {"range": [t_med * 100, t_high * 100], "color": "rgba(245,158,11,0.22)"},
                            {"range": [t_high * 100, t_crit * 100], "color": "rgba(239,68,68,0.22)"},
                            {"range": [t_crit * 100, 100], "color": "rgba(240,147,251,0.25)"},
                        ],
                    },
                ))
                st.plotly_chart(_plotly_dark(fig_g, 260), use_container_width=True)
            with r_col:
                st.markdown(f"### Risk Category: **{lvl} Risk** (`{prob:.2%}`)")
                st.caption("Prototype threshold classification for educator decision support only.")
                local_df, base_val = compute_local_shap(active_model, X_train, X_in, active_name, scaler)
                nl_bullets = generate_grounded_nl_explanation(local_df, top_n=4)
                for b in nl_bullets:
                    st.markdown(f"- {b}")

            xai1, xai2 = st.tabs(["🧬 SHAP Waterfall & Contributions", "📊 LIME Local Decision Rules"])
            with xai1:
                try:
                    fig_wf = create_shap_waterfall_figure(local_df, base_val, X_in, max_display=10)
                    st.pyplot(fig_wf, clear_figure=True)
                except Exception as e:
                    st.warning(f"SHAP waterfall notice: {e}")
            with xai2:
                try:
                    lime_exp = build_lime_explainer(X_train)
                    df_lime, lime_html = explain_instance_lime(lime_exp, active_model, X_in, active_name, scaler, num_features=10)
                    fig_l = px.bar(
                        df_lime, x="Weight", y="Rule", orientation="h", color="Direction",
                        color_discrete_map={"Increases Dropout Risk": "#f5576c", "Protective (Reduces Risk)": "#10b981"},
                        title="LIME Local Rules (Dropout = 1)",
                    )
                    fig_l.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(_plotly_dark(fig_l, 320), use_container_width=True)
                    with st.expander("🔍 Raw LIME HTML Explanation"):
                        components.html(lime_html, height=340, scrolling=True)
                except Exception as e:
                    st.warning(f"LIME notice: {e}")

            plan = generate_intervention_plan(X_in.iloc[0].to_dict(), prob, thresholds)
            st.markdown("#### 💡 Suggested Decision-Support Interventions")
            for sec, key in [("Academic", "academic"), ("Financial", "financial"), ("Engagement", "engagement")]:
                if plan[key]:
                    st.markdown(f"**{sec}:** " + " | ".join(plan[key]))
        st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # 6. REPORTS
    # =========================================================================
    elif st.session_state.navigation == "📈 Reports":
        st.markdown('<h1 class="main-title">📈 Reports & Export Center</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Generate student-level PDF dossiers, cohort risk distribution reports, and measured model performance exports.</p>', unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Cohort & Model Performance PDF")
            m_dict = comparison_df.loc[active_name].to_dict()
            rc_dict = risk_levels.value_counts().to_dict()
            cohort_pdf = build_cohort_summary_pdf(active_name, m_dict, rc_dict, len(X_test))
            st.download_button("📥 Download Cohort & Model PDF Report", data=cohort_pdf, file_name="eduguard_cohort_report.pdf", mime="application/pdf")
            st.markdown("</div>", unsafe_allow_html=True)
        with r2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Model Metrics CSV Export")
            st.download_button("📥 Download Model Comparison CSV", data=comparison_df.to_csv().encode("utf-8"), file_name="eduguard_model_metrics.csv", mime="text/csv")
            st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # 7. RESEARCH KNOWLEDGE BASE
    # =========================================================================
    elif st.session_state.navigation == "📚 Research Knowledge Base":
        st.markdown('<h1 class="main-title">📚 30-Paper Research Knowledge Base</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Local BM25 retrieval over all 30 verified PDFs in `papers_pdf/` with source paper and page attribution. Separate from ML training data.</p>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c_q1, c_q2 = st.columns([2, 1])
        with c_q1:
            preset = st.selectbox("Preset Research Questions:", ["(Custom Query)"] + PRESET_QUESTIONS)
            query_txt = st.text_input("Ask a question about the 30 papers:", value="" if preset == "(Custom Query)" else preset)
        with c_q2:
            cat_filter = st.selectbox("Filter by Category (14):", ["All"] + KNOWLEDGE_CATEGORIES)
            top_k = st.slider("Results:", 3, 10, 5)
        st.markdown("</div>", unsafe_allow_html=True)

        if query_txt.strip():
            hits = kb.query(query_txt, top_k=top_k, category_filter=cat_filter)
            st.write(f"Retrieved **{len(hits)}** matching papers:")
            for h in hits:
                with st.expander(f"Paper #{h['paper_num']:02d}: {h['title']} ({h['authors']}, {h['year']}) — [{h['page_ref']}] | Score: {h['score']}"):
                    st.markdown(f"**Source File:** `{h['filename']}` ({h['page_ref']}) | **Classification:** {h['relevance_type']}")
                    st.markdown(f"**Published Finding:** {h['published_finding']}")
                    st.markdown(f"**Reported Results:** {h['experimental_results']}")
                    st.markdown(f"**Limitations:** {h['limitations']}")
                    st.info(f"**EduGuard Implementation Note:** {h['eduguard_implementation_note']}")
        else:
            counts = kb.category_counts()
            st.dataframe(pd.DataFrame(list(counts.items()), columns=["Knowledge Category", "Indexed Papers"]), use_container_width=True)


if __name__ == "__main__":
    main()
