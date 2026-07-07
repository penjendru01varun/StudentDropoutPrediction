# ============= IMPORTS =============
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import os
import urllib.request
import zipfile
import io
import warnings
from fpdf import FPDF

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import shap
import lime
import lime.lime_tabular

warnings.filterwarnings('ignore')

# ============= CONFIGURATION =============
st.set_page_config(
    page_title="EduGuard | Student Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

TARGET_MAPPING = {
    'Dropout': 0,
    'Graduate': 1,
    'Enrolled': 2
}
REVERSE_TARGET_MAPPING = {0: 'Dropout', 1: 'Graduate', 2: 'Enrolled'}

# ============= CUSTOM CSS (PREMIUM GLASSMORPHISM DARK THEME) =============
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
        font-family: 'Poppins', 'Inter', sans-serif;
    }

    /* Glowing decorative backgrounds */
    .stApp::before {
        content: "";
        position: fixed;
        top: -10%;
        left: -10%;
        width: 40%;
        height: 40%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.12) 0%, rgba(0,0,0,0) 70%);
        z-index: 0;
        pointer-events: none;
    }
    .stApp::after {
        content: "";
        position: fixed;
        bottom: -10%;
        right: -10%;
        width: 50%;
        height: 50%;
        background: radial-gradient(circle, rgba(245, 87, 108, 0.1) 0%, rgba(0,0,0,0) 70%);
        z-index: 0;
        pointer-events: none;
    }

    /* Glassmorphism Card style */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.35);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        margin-bottom: 20px;
        z-index: 1;
        position: relative;
    }
    
    .glass-card:hover {
        transform: translateY(-4px) scale(1.01);
        border-color: rgba(118, 75, 162, 0.4);
        box-shadow: 0 12px 40px 0 rgba(118, 75, 162, 0.25);
    }

    /* KPI metric container */
    .kpi-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .kpi-icon {
        font-size: 2.2rem;
        padding: 12px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255, 255, 255, 0.05);
    }

    /* Custom headers */
    .main-title {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 50%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 5px;
        text-shadow: 0 0 30px rgba(240, 147, 251, 0.15);
    }
    
    .subtitle {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.15rem;
        margin-bottom: 30px;
        font-weight: 300;
    }

    .section-header {
        font-weight: 700;
        font-size: 1.5rem;
        color: #ffffff;
        border-left: 5px solid #764ba2;
        padding-left: 12px;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    /* Table custom styles */
    .dataframe {
        border-collapse: collapse;
        width: 100%;
        border-radius: 12px;
        overflow: hidden;
        background: rgba(255, 255, 255, 0.02) !important;
    }

    /* Risk badges styling and animations */
    .risk-badge {
        font-weight: 600;
        padding: 6px 12px;
        border-radius: 8px;
        text-align: center;
        display: inline-block;
    }

    @keyframes pulse-critical {
        0% { box-shadow: 0 0 0 0 rgba(118, 75, 162, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(118, 75, 162, 0); }
        100% { box-shadow: 0 0 0 0 rgba(118, 75, 162, 0); }
    }

    @keyframes pulse-high {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    
    .risk-critical {
        background: rgba(118, 75, 162, 0.25) !important;
        border: 1px solid #764ba2 !important;
        color: #f093fb !important;
        animation: pulse-critical 2s infinite;
    }
    
    .risk-high {
        background: rgba(239, 68, 68, 0.25) !important;
        border: 1px solid #ef4444 !important;
        color: #fee2e2 !important;
        animation: pulse-high 2s infinite;
    }
    
    .risk-medium {
        background: rgba(245, 158, 11, 0.25) !important;
        border: 1px solid #f59e0b !important;
        color: #fef3c7 !important;
    }
    
    .risk-low {
        background: rgba(16, 185, 129, 0.25) !important;
        border: 1px solid #10b981 !important;
        color: #d1fae5 !important;
    }

    /* Gradient buttons styling */
    div.stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 28px !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.4) !important;
        transition: all 0.3s ease !important;
        cursor: pointer;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(118, 75, 162, 0.6) !important;
        border: none !important;
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
    }

    div.stButton > button:active {
        transform: translateY(1px) !important;
    }

    /* Input widgets customization */
    .stSlider [data-baseweb="slider"] {
        background-color: transparent !important;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.25);
    }
</style>
""", unsafe_allow_html=True)


# ============= DATA LOADING & PREPROCESSING =============
def ensure_dataset():
    """Download and extract dataset from UCI repository if missing"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(base_dir, "data", "student_dropout_data.csv")
    
    if os.path.exists(local_path):
        return local_path
    
    # Check alternate locations
    alt_paths = [
        os.path.join(base_dir, 'student_dropout_data.csv'),
        os.path.join(base_dir, 'dataset.csv'),
        os.path.join(os.path.dirname(base_dir), 'data', 'student_dropout_data.csv')
    ]
    for path in alt_paths:
        if os.path.exists(path):
            return path

    # Download dataset from UCI Repository
    try:
        st.info("🌐 Fetching UCI Student Dropout dataset from archive...")
        url = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
        
        response = urllib.request.urlopen(url)
        zip_file = zipfile.ZipFile(io.BytesIO(response.read()))
        
        for name in zip_file.namelist():
            if name.endswith('.csv'):
                data = zip_file.read(name)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, 'wb') as f:
                    f.write(data)
                st.success("✅ Dataset successfully downloaded and cached!")
                return local_path
    except Exception as e:
        st.error(f"❌ Failed to fetch dataset: {str(e)}")
        # Fallback Mock Data Generation
        st.warning("⚠️ Generating sample dataset for fallback...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        np.random.seed(42)
        df_mock = pd.DataFrame({
            'Marital status': np.random.choice([1,2,3], 200),
            'Application mode': np.random.randint(1, 18, 200),
            'Application order': np.random.randint(1, 6, 200),
            'Course': np.random.choice([9238, 9254, 9500], 200),
            'Daytime/evening attendance': np.random.choice([0,1], 200),
            'Previous qualification': np.random.randint(1, 18, 200),
            'Previous qualification (grade)': np.random.uniform(95.0, 190.0, 200),
            'Nacionality': np.ones(200, dtype=int),
            'Mother\'s qualification': np.random.randint(1, 20, 200),
            'Father\'s qualification': np.random.randint(1, 20, 200),
            'Mother\'s occupation': np.random.randint(1, 10, 200),
            'Father\'s occupation': np.random.randint(1, 10, 200),
            'Admission grade': np.random.uniform(95.0, 190.0, 200),
            'Displaced': np.random.choice([0,1], 200),
            'Educational special needs': np.random.choice([0,1], 200, p=[0.95, 0.05]),
            'Debtor': np.random.choice([0,1], 200, p=[0.8, 0.2]),
            'Tuition fees up to date': np.random.choice([0,1], 200, p=[0.1, 0.9]),
            'Gender': np.random.choice([0,1], 200),
            'Scholarship holder': np.random.choice([0,1], 200, p=[0.75, 0.25]),
            'Age at enrollment': np.random.randint(18, 55, 200),
            'International': np.random.choice([0,1], 200, p=[0.97, 0.03]),
            'Curricular units 1st sem (credited)': np.random.randint(0, 5, 200),
            'Curricular units 1st sem (enrolled)': np.random.randint(4, 10, 200),
            'Curricular units 1st sem (evaluations)': np.random.randint(4, 12, 200),
            'Curricular units 1st sem (approved)': np.random.randint(0, 10, 200),
            'Curricular units 1st sem (grade)': np.random.uniform(0.0, 20.0, 200),
            'Curricular units 1st sem (without evaluations)': np.random.randint(0, 3, 200),
            'Curricular units 2nd sem (credited)': np.random.randint(0, 5, 200),
            'Curricular units 2nd sem (enrolled)': np.random.randint(4, 10, 200),
            'Curricular units 2nd sem (evaluations)': np.random.randint(4, 12, 200),
            'Curricular units 2nd sem (approved)': np.random.randint(0, 10, 200),
            'Curricular units 2nd sem (grade)': np.random.uniform(0.0, 20.0, 200),
            'Curricular units 2nd sem (without evaluations)': np.random.randint(0, 3, 200),
            'Unemployment rate': np.random.uniform(7.0, 16.0, 200),
            'Inflation rate': np.random.uniform(-1.0, 4.0, 200),
            'GDP': np.random.uniform(-4.0, 4.0, 200),
            'Target': np.random.choice(['Graduate', 'Dropout', 'Enrolled'], 200, p=[0.5, 0.32, 0.18])
        })
        df_mock.to_csv(local_path, index=False)
        return local_path

@st.cache_data
def load_data(uploaded_file=None):
    """Load and preprocess dataset"""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            
    dataset_path = ensure_dataset()
    if dataset_path and os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path, sep=None, engine='python')
            return df
        except Exception as e:
            st.error(f"Error reading dataset file: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


# ============= FEATURE ENGINEERING =============
@st.cache_data
def engineer_features(df):
    """Perform Feature Engineering for Early Warning System"""
    df_feat = df.copy()
    
    # 1. Credit Completion Rate 1st Sem = approved / enrolled
    if 'Curricular units 1st sem (approved)' in df_feat.columns and 'Curricular units 1st sem (enrolled)' in df_feat.columns:
        df_feat['Credit_Completion_Rate_1st_Sem'] = (
            df_feat['Curricular units 1st sem (approved)'] / 
            df_feat['Curricular units 1st sem (enrolled)'].replace(0, 1)
        ).fillna(0.0)
    else:
        df_feat['Credit_Completion_Rate_1st_Sem'] = 1.0
        
    # 2. Credit Completion Rate 2nd Sem = approved / enrolled
    if 'Curricular units 2nd sem (approved)' in df_feat.columns and 'Curricular units 2nd sem (enrolled)' in df_feat.columns:
        df_feat['Credit_Completion_Rate_2nd_Sem'] = (
            df_feat['Curricular units 2nd sem (approved)'] / 
            df_feat['Curricular units 2nd sem (enrolled)'].replace(0, 1)
        ).fillna(0.0)
    else:
        df_feat['Credit_Completion_Rate_2nd_Sem'] = 1.0
        
    # 3. Grade Drop = 1st_sem_grade - 2nd_sem_grade
    if 'Curricular units 1st sem (grade)' in df_feat.columns and 'Curricular units 2nd sem (grade)' in df_feat.columns:
        df_feat['Grade_Drop'] = (
            df_feat['Curricular units 1st sem (grade)'] - 
            df_feat['Curricular units 2nd sem (grade)']
        ).fillna(0.0)
    else:
        df_feat['Grade_Drop'] = 0.0
        
    # 4. Engagement Score = ((evaluations_1st / enrolled_1st) + (evaluations_2nd / enrolled_2nd)) / 2
    if ('Curricular units 1st sem (evaluations)' in df_feat.columns and 
        'Curricular units 2nd sem (evaluations)' in df_feat.columns and 
        'Curricular units 1st sem (enrolled)' in df_feat.columns and 
        'Curricular units 2nd sem (enrolled)' in df_feat.columns):
        
        eng_1st = df_feat['Curricular units 1st sem (evaluations)'] / df_feat['Curricular units 1st sem (enrolled)'].replace(0, 1)
        eng_2nd = df_feat['Curricular units 2nd sem (evaluations)'] / df_feat['Curricular units 2nd sem (enrolled)'].replace(0, 1)
        df_feat['Engagement_Score'] = ((eng_1st + eng_2nd) / 2.0).fillna(0.0)
    else:
        df_feat['Engagement_Score'] = 1.0
        
    # 5. Academic Stress Index = enrolled - approved
    if ('Curricular units 1st sem (enrolled)' in df_feat.columns and 
        'Curricular units 2nd sem (enrolled)' in df_feat.columns and 
        'Curricular units 1st sem (approved)' in df_feat.columns and 
        'Curricular units 2nd sem (approved)' in df_feat.columns):
        
        total_enrolled = df_feat['Curricular units 1st sem (enrolled)'] + df_feat['Curricular units 2nd sem (enrolled)']
        total_approved = df_feat['Curricular units 1st sem (approved)'] + df_feat['Curricular units 2nd sem (approved)']
        df_feat['Academic_Stress_Index'] = (total_enrolled - total_approved).fillna(0.0)
    else:
        df_feat['Academic_Stress_Index'] = 0.0
        
    # 6. Financial Risk = Debtor + (1 - Tuition fees up to date) + (1 - Scholarship holder)
    debtor = df_feat['Debtor'] if 'Debtor' in df_feat.columns else 0.0
    tuition = df_feat['Tuition fees up to date'] if 'Tuition fees up to date' in df_feat.columns else 1.0
    scholarship = df_feat['Scholarship holder'] if 'Scholarship holder' in df_feat.columns else 0.0
    df_feat['Financial_Risk'] = (debtor + (1.0 - tuition) + (1.0 - scholarship)).fillna(0.0)
    
    return df_feat


def preprocess_data(df):
    """Preprocess dataset for model training"""
    processed_df = df.copy()
    if 'id' in processed_df.columns:
        processed_df = processed_df.drop('id', axis=1)
        
    processed_df = processed_df.fillna(processed_df.median(numeric_only=True))
    
    # Label encode categorical object columns if any
    le = LabelEncoder()
    categorical_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'Target']
    for col in categorical_cols:
        processed_df[col] = le.fit_transform(processed_df[col])
        
    # Map target variable
    if 'Target' in processed_df.columns:
        if processed_df['Target'].dtype == 'object':
            processed_df['Target'] = processed_df['Target'].map(TARGET_MAPPING)
        processed_df['Target'] = processed_df['Target'].fillna(0).astype(int)
        
    for col in processed_df.columns:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
    processed_df.dropna(inplace=True)
    
    # XGBoost/LightGBM clean brackets from columns
    cleaned_cols = [c.replace('[', '').replace(']', '').replace('<', '') for c in processed_df.columns]
    processed_df.columns = cleaned_cols
    
    X = processed_df.drop('Target', axis=1)
    y = processed_df['Target']
    
    return X, y, processed_df


def preprocess_inference_data(df, training_columns):
    """Preprocess batch files or predictor data (which may not contain a 'Target' column)"""
    processed_df = df.copy()
    if 'id' in processed_df.columns:
        processed_df = processed_df.drop('id', axis=1)
    if 'Target' in processed_df.columns:
        processed_df = processed_df.drop('Target', axis=1)
        
    processed_df = processed_df.fillna(processed_df.median(numeric_only=True))
    
    # Label encode categorical object columns if any
    le = LabelEncoder()
    categorical_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        processed_df[col] = le.fit_transform(processed_df[col])
        
    for col in processed_df.columns:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
    # Clean brackets from columns to match model
    cleaned_cols = [c.replace('[', '').replace(']', '').replace('<', '') for c in processed_df.columns]
    processed_df.columns = cleaned_cols
    
    # Align columns with training features
    for col in training_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0.0
            
    processed_df = processed_df[training_columns]
    return processed_df


# ============= MODEL TRAINING =============
@st.cache_resource
def train_and_compare_models(df_engineered):
    """Train models with SMOTE and calculate evaluation metrics"""
    X, y, processed_df = preprocess_data(df_engineered)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # SMOTE to handle imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Standardize for Logistic Regression
    scaler = StandardScaler()
    X_train_res_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1),
        'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    }
    
    results = {}
    fitted_models = {}
    
    for name, model in models.items():
        if name in ['Logistic Regression']:
            model.fit(X_train_res_scaled, y_train_res)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)
        else:
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            
        fitted_models[name] = model
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        try:
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        except Exception:
            auc = 0.0
            
        results[name] = {
            'Accuracy': acc,
            'Precision': report['macro avg']['precision'],
            'Recall': report['macro avg']['recall'],
            'F1-Score': report['macro avg']['f1-score'],
            'ROC_AUC': auc
        }
        
    comparison_df = pd.DataFrame(results).T
    
    return fitted_models, comparison_df, X_train, X_test, y_train, y_test, X_train_res, y_train_res, scaler


# ============= EXPLAINABILITY UTILITIES =============
def get_class_shap_values(shap_values, class_idx):
    """Retrieve SHAP values for a specific class depending on format"""
    if isinstance(shap_values, list):
        return shap_values[class_idx]
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        return shap_values[:, :, class_idx]
    else:
        return shap_values


@st.cache_resource
def get_lime_explainer(_X_train):
    """LIME Explainer initialized on training background"""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(_X_train),
        feature_names=_X_train.columns.tolist(),
        class_names=['Dropout', 'Graduate', 'Enrolled'],
        mode='classification',
        random_state=42
    )
    return explainer


@st.cache_resource
def get_shap_explainer(_model, _X_train, model_name, _scaler=None):
    """SHAP Explainer initialized on training background"""
    try:
        if model_name == 'Logistic Regression':
            scaled_bg = _scaler.transform(_X_train.sample(n=min(100, len(_X_train)), random_state=42))
            explainer = shap.LinearExplainer(_model, scaled_bg)
        elif model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
            explainer = shap.TreeExplainer(_model)
        else:
            explainer = shap.Explainer(_model, _X_train.sample(n=100, random_state=42))
        return explainer
    except Exception as e:
        st.warning(f"Could not create SHAP explainer: {e}")
        return None


def generate_natural_language_explanation(row, shap_values_dict):
    """Generate professional, educator-friendly explanation of student risk"""
    sorted_features = sorted(shap_values_dict.items(), key=lambda x: x[1], reverse=True)
    
    bullet_points = []
    count = 1
    
    for feat, shap_val in sorted_features:
        if shap_val <= 0.005:
            continue
        val = row.get(feat, 0.0)
        
        if 'Curricular units 1st sem (approved)' in feat or 'Credit_Completion_Rate_1st_Sem' in feat:
            bullet_points.append(f"🎓 **Academic Performance (1st Sem)**: Low credit completion. The student approved only {row.get('Curricular units 1st sem (approved)', 0):.0f} of their enrolled units. Historically, a failure to complete credits in the first semester is the single strongest indicator of long-term disengagement.")
        elif 'Curricular units 2nd sem (approved)' in feat or 'Credit_Completion_Rate_2nd_Sem' in feat:
            bullet_points.append(f"📚 **Academic Progress (2nd Sem)**: Only {row.get('Curricular units 2nd sem (approved)', 0):.0f} units completed in the second semester. Difficulty in program progression indicates a high risk of academic overload.")
        elif 'Tuition fees up to date' in feat and val == 0:
            bullet_points.append(f"💳 **Tuition Overdue**: Tuition fees are not up to date, adding substantial financial stress and placing institutional registration holds at risk.")
        elif 'Debtor' in feat and val == 1:
            bullet_points.append(f"⚠️ **Institutional Debt**: Outstanding debt increases the probability of dropout due to sudden financial hardship and hold policies.")
        elif 'Scholarship holder' in feat and val == 0:
            bullet_points.append(f"💸 **No Financial Scholarship**: Lack of a scholarship/bursary increases reliance on external employment or loans, leading to exhaustion and stress.")
        elif 'Academic_Stress_Index' in feat and val > 2.0:
            bullet_points.append(f"🧠 **High Academic Stress**: An stress index of {val:.1f} credits represents an overloaded class schedule relative to credit success, leading to fatigue.")
        elif 'Financial_Risk' in feat and val > 1.0:
            bullet_points.append(f"💰 **Financial Risk Score**: The financial vulnerability index is {val:.1f}/3.0. Outstanding debt combined with lack of funding makes continued enrollment highly unstable.")
        elif 'Grade_Drop' in feat and val > 1.5:
            bullet_points.append(f"📉 **Declining Grades**: Average grade dropped by {val:.1f} points between semesters, showing sudden academic difficulty or disengagement.")
        elif 'Curricular units 1st sem (grade)' in feat and val < 10.0:
            bullet_points.append(f"✏️ **Low 1st Sem Grade**: Average grade is {val:.1f}/20, indicating immediate academic struggles in foundational courses.")
        elif 'Age at enrollment' in feat and val > 25:
            bullet_points.append(f"👤 **Non-Traditional Age**: Enrolling at an older age ({val:.0f} yrs) correlates with external responsibilities (family, work) that complicate scheduling.")
        else:
            feat_name = feat.replace('_', ' ').title()
            bullet_points.append(f"🔍 **{feat_name}**: Feature value ({val:.2f}) had a positive correlation with dropout risk in this model run.")
            
        count += 1
        if count > 4:
            break
            
    if not bullet_points:
        return "No major risk drivers identified. The student's academic and financial indicators are generally supportive and low-risk."
        
    return "\n\n".join(bullet_points)


def get_actionable_recommendations(row, risk_proba):
    """Generate personalized recommendation steps based on student features"""
    recs = []
    
    if risk_proba < 0.40:
        recs.append("✅ **Regular Monitoring**: No immediate intervention needed. Keep the student engaged with routine check-ins and standard academic advising.")
        return recs
        
    # Academic interventions
    first_sem_app = row.get('Curricular units 1st sem (approved)', 6)
    first_sem_grade = row.get('Curricular units 1st sem (grade)', 12.0)
    grade_drop = row.get('Grade_Drop', 0.0)
    stress_idx = row.get('Academic_Stress_Index', 0.0)
    
    if first_sem_app <= 2 or first_sem_grade < 10.0:
        recs.append("📚 **Academic Coaching**: Schedule mandatory meetings with an academic advisor. Recommend enrolling in foundational tutoring classes and peer study groups.")
    if grade_drop > 2.0:
        recs.append("📈 **Performance Review**: A significant grade decline is observed. Contact the student to identify sudden personal, health, or adjustment issues.")
    if stress_idx > 4.0:
        recs.append("🧠 **Credit Load Adjustment**: The academic stress index is high. Consider counseling the student to adjust their credit load or withdraw from non-essential electives.")
        
    # Financial interventions
    debtor = row.get('Debtor', 0)
    tuition = row.get('Tuition fees up to date', 1)
    scholarship = row.get('Scholarship holder', 0)
    fin_risk = row.get('Financial_Risk', 0.0)
    
    if debtor == 1 or tuition == 0:
        recs.append("💰 **Financial Counseling**: Connect the student with the Financial Aid Office immediately. Explore tuition payment restructuring or emergency student bursaries.")
    elif (scholarship == 0 and risk_proba > 0.60) or fin_risk > 1.5:
        recs.append("💸 **Scholarship Support**: Help the student search for local grants, work-study options, or need-based scholarships.")
        
    # Stress & Demographic interventions
    displaced = row.get('Displaced', 0)
    
    if displaced == 1:
        recs.append("🏠 **Social Integration Support**: Connect the student with campus groups, commuter student clubs, or residential mentorship panels.")
        
    if not recs:
        recs.append("🔍 **Comprehensive Advising Interview**: Set up a face-to-face academic review to identify non-quantified hurdles.")
        
    return recs


# ============= EXPORT FUNCTIONS =============
def generate_student_pdf_report(student_id, student_row, risk_proba, risk_level, recommendations, why_at_risk):
    """Generate high-quality PDF report for student intervention"""
    pdf = FPDF()
    pdf.add_page()
    
    def c_txt(t):
        if not isinstance(t, str):
            t = str(t)
        cleaned = t.replace('**', '').replace('###', '').replace('##', '').replace('#', '')
        for char in ['🎓', '📚', '💳', '⚠️', '💸', '🧠', '📉', '✏️', '🔍', '✅', '📈', '💰', '🏠', '🔴', '🟡', '🟢', '🟣', '⚡']:
            cleaned = cleaned.replace(char, '-')
        return cleaned.encode('latin-1', 'replace').decode('latin-1')

    # Deep Space/Purple Theme Colors
    pdf.set_fill_color(15, 12, 41)
    pdf.set_text_color(255, 255, 255)
    
    # Title Banner
    pdf.rect(0, 0, 210, 40, "F")
    pdf.set_y(12)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 8, c_txt("STUDENT EARLY WARNING & INTERVENTION REPORT"), 0, 1, "C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, c_txt("Generated by EduGuard Early Warning System | Predictive Insights"), 0, 1, "C")
    
    # Content Reset
    pdf.set_text_color(51, 65, 85)
    pdf.set_y(48)
    
    # Section: Student Details
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(48, 43, 99)
    pdf.cell(0, 8, c_txt("1. Student Profile & Metadata"), 0, 1, "L")
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(71, 85, 105)
    
    metadata_fields = [
        ("Student ID", f"#{student_id}"),
        ("Age at Enrollment", f"{student_row.get('Age at enrollment', 'N/A')} yrs"),
        ("Gender", "Male" if student_row.get('Gender', 0) == 1 else "Female"),
        ("Scholarship Holder", "Yes" if student_row.get('Scholarship holder', 0) == 1 else "No"),
        ("Tuition Fees Up to Date", "Yes" if student_row.get('Tuition fees up to date', 0) == 1 else "No"),
        ("Institutional Debtor", "Yes" if student_row.get('Debtor', 0) == 1 else "No"),
        ("Course Code", f"{student_row.get('Course', 'N/A')}"),
        ("Admission Grade", f"{student_row.get('Admission grade', 'N/A')}/200"),
        ("Academic Stress Index", f"{student_row.get('Academic_Stress_Index', 0.0):.2f}"),
        ("Grade Drop (S1 - S2)", f"{student_row.get('Grade_Drop', 0.0):.2f}")
    ]
    
    for i in range(0, len(metadata_fields), 2):
        k1, v1 = metadata_fields[i]
        k2, v2 = metadata_fields[i+1] if i+1 < len(metadata_fields) else ("", "")
        pdf.cell(45, 7, f"  {c_txt(k1)}:", 1, 0, "L")
        pdf.cell(50, 7, f"  {c_txt(v1)}", 1, 0, "L")
        pdf.cell(45, 7, f"  {c_txt(k2)}:", 1, 0, "L")
        pdf.cell(50, 7, f"  {c_txt(v2)}", 1, 1, "L")
        
    pdf.ln(5)
    
    # Section: Risk Assessment
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(48, 43, 99)
    pdf.cell(0, 8, c_txt("2. AI Risk Assessment Matrix"), 0, 1, "L")
    
    if risk_level == "Critical":
        pdf.set_fill_color(118, 75, 162)
        text_color = (255, 255, 255)
    elif risk_level == "High":
        pdf.set_fill_color(239, 68, 68)
        text_color = (255, 255, 255)
    elif risk_level == "Medium":
        pdf.set_fill_color(245, 158, 11)
        text_color = (0, 0, 0)
    else:
        pdf.set_fill_color(16, 185, 129)
        text_color = (255, 255, 255)
        
    pdf.set_text_color(*text_color)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(190, 10, f" RISK LEVEL: {risk_level.upper()} RISK  (Dropout Probability: {risk_proba:.2%})", 0, 1, "C", True)
    
    pdf.ln(5)
    pdf.set_text_color(71, 85, 105)
    
    # Section: Why At Risk
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(48, 43, 99)
    pdf.cell(0, 8, c_txt("3. Explainable AI Risk Drivers"), 0, 1, "L")
    pdf.set_font("Arial", "", 9.5)
    pdf.set_text_color(51, 65, 85)
    
    pdf.multi_cell(0, 5, c_txt(why_at_risk))
    pdf.ln(5)
    
    # Section: Recommendations
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(48, 43, 99)
    pdf.cell(0, 8, c_txt("4. Targeted Action Plan"), 0, 1, "L")
    pdf.set_font("Arial", "", 9.5)
    pdf.set_text_color(51, 65, 85)
    
    for rec in recommendations:
        pdf.multi_cell(0, 5, f"- {c_txt(rec)}")
        
    pdf.ln(10)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 8, c_txt("Confidential Dossier. Designed for internal academic advising use only."), 0, 0, "C")
    
    return pdf.output()


# ============= MAIN APP =============
def main():
    # Load dataset
    df = load_data()
    if df.empty:
        st.error("No dataset available. Please ensure data is loaded properly.")
        return
        
    # Engineer features
    df_engineered = engineer_features(df)
    
    # Train models and compare
    (fitted_models, comparison_df, X_train, X_test, 
     y_train, y_test, X_train_res, y_train_res, scaler) = train_and_compare_models(df_engineered)
    
    # Session State Initialization
    if 'selected_student_id' not in st.session_state:
        st.session_state.selected_student_id = None
    if 'selected_model_name' not in st.session_state:
        st.session_state.selected_model_name = 'Random Forest'
    if 'navigation' not in st.session_state:
        st.session_state.navigation = "🏠 Dashboard"

    def switch_page(page_name):
        st.session_state.navigation = page_name
        st.rerun()

    # SIDEBAR NAVIGATION
    st.sidebar.markdown("""
        <div style="text-align: center; margin-bottom: 25px;">
            <h2 style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.2rem; margin: 0;">🎓 EduGuard</h2>
            <p style="color: rgba(255,255,255,0.6); font-size: 0.85rem; margin-top: 5px; text-transform: uppercase; letter-spacing: 1px;">Early Warning Dashboard</p>
        </div>
    """, unsafe_allow_html=True)
    
    menu = [
        "🏠 Dashboard", 
        "📊 Data Explorer", 
        "🤖 Model Hub", 
        "⚠️ Early Warning System", 
        "🔮 Predictor", 
        "📈 Reports"
    ]
    
    # Stateful navigation radio
    active_idx = menu.index(st.session_state.navigation) if st.session_state.navigation in menu else 0
    choice = st.sidebar.radio("Select Navigation", menu, index=active_idx, key="nav_radio", label_visibility="collapsed")
    if choice != st.session_state.navigation:
        st.session_state.navigation = choice
        st.rerun()
        
    st.sidebar.markdown("---")
    
    # Global Model Selector
    st.sidebar.markdown("### 🤖 Active Prediction Model")
    st.session_state.selected_model_name = st.sidebar.selectbox(
        "Choose Model:", 
        list(fitted_models.keys()), 
        index=list(fitted_models.keys()).index(st.session_state.selected_model_name)
    )
    
    # Global Controls
    st.sidebar.markdown("### ⚙️ Upload Custom Data")
    uploaded_file = st.sidebar.file_uploader("Upload custom CSV", type=['csv'])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df_engineered = engineer_features(df)
        (fitted_models, comparison_df, X_train, X_test, 
         y_train, y_test, X_train_res, y_train_res, scaler) = train_and_compare_models(df_engineered)
        st.sidebar.success("Custom Dataset Loaded!")
        
    st.sidebar.markdown("---")
    st.sidebar.info("💡 Version 2.5.0 (Early Warning Engine)")

    # Probabilities calculation based on active model
    model = fitted_models[st.session_state.selected_model_name]
    if st.session_state.selected_model_name == 'Logistic Regression':
        probs = model.predict_proba(scaler.transform(X_test))[:, 0]
    else:
        probs = model.predict_proba(X_test)[:, 0]

    # 1. DASHBOARD PAGE
    if st.session_state.navigation == "🏠 Dashboard":
        st.markdown('<h1 class="main-title">🎓 Student Dropout Early Warning System</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Real-time predictive analytics platform to identify and mitigate academic dropout risk.</p>', unsafe_allow_html=True)
        
        # Calculate Key Metrics
        total_students = len(df_engineered)
        
        if df_engineered['Target'].dtype in [int, np.int64]:
            dropout_count = (df_engineered['Target'] == 0).sum()
            grad_count = (df_engineered['Target'] == 1).sum()
        else:
            dropout_count = (df_engineered['Target'] == 'Dropout').sum()
            grad_count = (df_engineered['Target'] == 'Graduate').sum()
            
        dropout_rate = (dropout_count / total_students) * 100
        grad_rate = (grad_count / total_students) * 100
        
        critical_count = (probs >= 0.90).sum()
        high_count = ((probs >= 0.70) & (probs < 0.90)).sum()
        
        # Metric Cards with glass styling
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(f"""
                <div class="glass-card kpi-container">
                    <div>
                        <p style="margin:0; font-size: 0.9rem; color: rgba(255,255,255,0.6)">Total Students</p>
                        <h2 style="margin:5px 0 0 0; font-size: 2.2rem; font-weight:800;">{total_students:,}</h2>
                    </div>
                    <div class="kpi-icon" style="background: rgba(102, 126, 234, 0.15); color: #667eea;">👥</div>
                </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
                <div class="glass-card kpi-container">
                    <div>
                        <p style="margin:0; font-size: 0.9rem; color: rgba(255,255,255,0.6)">Dropout Rate</p>
                        <h2 style="margin:5px 0 0 0; font-size: 2.2rem; font-weight:800; color: #f5576c;">{dropout_rate:.1f}%</h2>
                    </div>
                    <div class="kpi-icon" style="background: rgba(245, 87, 108, 0.15); color: #f5576c;">📈</div>
                </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
                <div class="glass-card kpi-container">
                    <div>
                        <p style="margin:0; font-size: 0.9rem; color: rgba(255,255,255,0.6)">Critical Risk (>=90%)</p>
                        <h2 style="margin:5px 0 0 0; font-size: 2.2rem; font-weight:800; color: #f093fb;">{critical_count}</h2>
                    </div>
                    <div class="kpi-icon" style="background: rgba(240, 147, 251, 0.15); color: #f093fb;">🚨</div>
                </div>
            """, unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"""
                <div class="glass-card kpi-container">
                    <div>
                        <p style="margin:0; font-size: 0.9rem; color: rgba(255,255,255,0.6)">High Risk (70-89%)</p>
                        <h2 style="margin:5px 0 0 0; font-size: 2.2rem; font-weight:800; color: #fee2e2;">{high_count}</h2>
                    </div>
                    <div class="kpi-icon" style="background: rgba(254, 226, 226, 0.15); color: #fee2e2;">⚠️</div>
                </div>
            """, unsafe_allow_html=True)
            
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3 style="margin:0 0 15px 0; font-size: 1.25rem;">📊 Real-time Risk Distribution</h3>', unsafe_allow_html=True)
            
            risk_series = pd.Series(probs).apply(
                lambda p: 'Critical (>=90%)' if p >= 0.90 else ('High (70-89%)' if p >= 0.70 else ('Medium (40-69%)' if p >= 0.40 else 'Low (<40%)'))
            )
            risk_counts = risk_series.value_counts().reset_index()
            risk_counts.columns = ['Risk Level', 'Count']
            
            fig = px.pie(
                risk_counts, 
                values='Count', 
                names='Risk Level',
                color='Risk Level',
                color_discrete_map={
                    'Critical (>=90%)': '#764ba2',
                    'High (70-89%)': '#ef4444',
                    'Medium (40-69%)': '#f59e0b',
                    'Low (<40%)': '#10b981'
                },
                hole=0.4
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff',
                margin=dict(t=10, b=10, l=10, r=10),
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_right:
            st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown('<h3 style="margin:0 0 15px 0; font-size: 1.25rem;">✨ Core Retention Innovations</h3>', unsafe_allow_html=True)
            st.write("""
            Our **Early Warning System (EWS)** predicts student dropout risk before disengagement occurs.
            By incorporating 6 unique composite features, the dashboard flags critical outliers and links explanations to tailored action plans.
            """)
            st.write("**Innovative features added:**")
            st.markdown("""
            - 📈 **Grade Drop Tracker**
            - 💳 **Financial Vulnerability Score**
            - 🧠 **Academic Stress Index**
            - 🔬 **SHAP & LIME Interpretability**
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Recent High-Risk Students List
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="margin:0 0 20px 0; font-size: 1.25rem;">🚨 High Risk Students requiring immediate mediation</h3>', unsafe_allow_html=True)
        
        test_summary = X_test.copy()
        test_summary['Dropout_Probability'] = probs
        test_summary['Risk_Level'] = test_summary['Dropout_Probability'].apply(
            lambda p: 'Critical' if p >= 0.90 else ('High' if p >= 0.70 else ('Medium' if p >= 0.40 else 'Low'))
        )
        test_summary['Student_ID'] = test_summary.index + 10000
        
        # Select Critical/High
        critical_df = test_summary[test_summary['Risk_Level'].isin(['Critical', 'High'])].sort_values('Dropout_Probability', ascending=False).head(5)
        
        if len(critical_df) > 0:
            for idx, row in critical_df.iterrows():
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f"""
                        <div style="padding: 14px 20px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-weight: 700; color: #ffffff; font-size: 1.05rem;">Student #{row['Student_ID']}</span>
                                <span style="margin-left: 20px; color: rgba(255,255,255,0.6); font-size: 0.9rem;">Course Code: {row['Course']:.0f}</span>
                                <span style="margin-left: 20px; color: rgba(255,255,255,0.6); font-size: 0.9rem;">Stress Index: {row['Academic_Stress_Index']:.1f}</span>
                                <span style="margin-left: 20px; color: rgba(255,255,255,0.6); font-size: 0.9rem;">Grade Drop: {row['Grade_Drop']:.1f}</span>
                            </div>
                            <div>
                                <span class="risk-badge risk-{row['Risk_Level'].lower()}">{row['Risk_Level']} ({row['Dropout_Probability']:.1%})</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                with c2:
                    if st.button(f"🔍 Intervene", key=f"dash_int_{row['Student_ID']}"):
                        st.session_state.selected_student_id = row['Student_ID']
                        switch_page("⚠️ Early Warning System")
        else:
            st.info("No active students fall in Critical or High Risk levels.")
            
        st.markdown('</div>', unsafe_allow_html=True)


    # 2. DATA EXPLORER PAGE
    elif st.session_state.navigation == "📊 Data Explorer":
        st.markdown('<h1 class="main-title">📊 Exploratory Data Analysis</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Investigate correlation trends, student demographics, and the impacts of composite features.</p>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["👥 Student Demographics", "📚 Academic Performance", "⚠️ Risk Co-factors", "🔍 Interactive Plotter"])
        
        with tab1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                fig_age = px.histogram(df_engineered, x="Age at enrollment", title="Age at Enrollment Distribution", color_discrete_sequence=['#764ba2'])
                fig_age.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
                st.plotly_chart(fig_age, use_container_width=True)
            with col2:
                gender_df = df_engineered['Gender'].map({0: 'Female', 1: 'Male'}).value_counts().reset_index()
                gender_df.columns = ['Gender', 'Count']
                fig_gen = px.pie(gender_df, values='Count', names='Gender', title="Gender Distribution", color_discrete_sequence=['#667eea', '#f5576c'])
                fig_gen.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
                st.plotly_chart(fig_gen, use_container_width=True)
            with col3:
                schol_df = df_engineered['Scholarship holder'].map({0: 'No Scholarship', 1: 'Scholarship Holder'}).value_counts().reset_index()
                schol_df.columns = ['Scholarship Status', 'Count']
                fig_schol = px.pie(schol_df, values='Count', names='Scholarship Status', title="Scholarship Status", color_discrete_sequence=['#10b981', '#f5576c'])
                fig_schol.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
                st.plotly_chart(fig_schol, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tab2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                df_viz = df_engineered.copy()
                if df_viz['Target'].dtype in [int, np.int64]:
                    df_viz['Target'] = df_viz['Target'].map(REVERSE_TARGET_MAPPING)
                fig_box1 = px.box(df_viz, x="Target", y="Curricular units 1st sem (grade)", title="1st Sem average grade vs Outcome", color="Target")
                fig_box1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
                st.plotly_chart(fig_box1, use_container_width=True)
            with col2:
                fig_box2 = px.box(df_viz, x="Target", y="Grade_Drop", title="Grade Drop (S1 - S2) vs Outcome", color="Target")
                fig_box2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
                st.plotly_chart(fig_box2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tab3:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                debt_cross = pd.crosstab(df_engineered['Debtor'], df_engineered['Target']).reset_index()
                debt_cross['Debtor'] = debt_cross['Debtor'].map({0: 'No Debt', 1: 'Active Debt'})
                if 0 in debt_cross.columns:
                    debt_cross.rename(columns={0: 'Dropout', 1: 'Graduate', 2: 'Enrolled'}, inplace=True)
                fig_debt = px.bar(debt_cross, x='Debtor', y=['Dropout', 'Graduate', 'Enrolled'], title="Dropout Breakdown by Debt Status", barmode='group')
                fig_debt.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
                st.plotly_chart(fig_debt, use_container_width=True)
            with col2:
                stress_box = px.box(df_viz, x="Target", y="Academic_Stress_Index", title="Academic Stress Index vs Outcome", color="Target")
                stress_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
                st.plotly_chart(stress_box, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tab4:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h4 style="margin:0 0 15px 0;">Interactive Feature Explorer</h4>', unsafe_allow_html=True)
            feat_list = [c for c in df_engineered.columns if c != 'Target']
            
            c_sel1, c_sel2 = st.columns(2)
            with c_sel1:
                x_feat = st.selectbox("Select X-Axis Feature:", feat_list, index=min(19, len(feat_list)-1))
            with c_sel2:
                y_feat = st.selectbox("Select Y-Axis Feature:", feat_list, index=min(24, len(feat_list)-1))
                
            fig_scatter = px.scatter(df_viz, x=x_feat, y=y_feat, color="Target", opacity=0.7, hover_data=['Age at enrollment'])
            fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


    # 3. MODEL HUB PAGE
    elif st.session_state.navigation == "🤖 Model Hub":
        st.markdown('<h1 class="main-title">🤖 Model Performance Hub</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Evaluate and compare prediction architectures. Visualise confusion matrices and global explanations.</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="margin:0 0 15px 0; font-size: 1.25rem;">📈 Performance Table (Macro Averages)</h3>', unsafe_allow_html=True)
        st.dataframe(comparison_df.style.background_gradient(cmap='Purples', subset=['Accuracy', 'F1-Score', 'ROC_AUC']), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3 style="margin:0 0 15px 0; font-size: 1.25rem;">📊 Multi-Class ROC Curves</h3>', unsafe_allow_html=True)
            
            # ROC Curves
            eval_model = fitted_models[st.session_state.selected_model_name]
            if st.session_state.selected_model_name == 'Logistic Regression':
                test_probs = eval_model.predict_proba(scaler.transform(X_test))
            else:
                test_probs = eval_model.predict_proba(X_test)
                
            fig_roc = go.Figure()
            for cls_id, cls_name in REVERSE_TARGET_MAPPING.items():
                y_test_bin = (y_test == cls_id).astype(int)
                fpr, tpr, _ = roc_curve(y_test_bin, test_probs[:, cls_id])
                auc_score = roc_auc_score(y_test_bin, test_probs[:, cls_id])
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{cls_name} (AUC={auc_score:.2f})'))
                
            fig_roc.add_shape(type='line', line=dict(dash='dash', color='rgba(255,255,255,0.4)'), x0=0, x1=1, y0=0, y1=1)
            fig_roc.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff',
                height=300,
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3 style="margin:0 0 15px 0; font-size: 1.25rem;">🧩 Confusion Matrix</h3>', unsafe_allow_html=True)
            
            # Confusion Matrix Heatmap
            if st.session_state.selected_model_name == 'Logistic Regression':
                y_pred = eval_model.predict(scaler.transform(X_test))
            else:
                y_pred = eval_model.predict(X_test)
                
            cm = confusion_matrix(y_test, y_pred)
            class_names = ['Dropout', 'Graduate', 'Enrolled']
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=class_names,
                y=class_names,
                colorscale='Purples',
                text=cm,
                texttemplate="%{text}",
                hoverongaps=False
            ))
            fig_cm.update_layout(
                xaxis_title="Predicted Class",
                yaxis_title="True Class",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff',
                height=300,
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Global Feature Importance section
        st.markdown('<h2 class="section-header">🌍 Global Feature Importance & SHAP Values</h2>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        rf_model = fitted_models['Random Forest']
        sample_size = min(100, len(X_train))
        X_sample = X_train.sample(n=sample_size, random_state=42)
        
        with st.spinner("🔄 Running SHAP TreeExplainer (sampled)..."):
            try:
                explainer = shap.TreeExplainer(rf_model)
                shap_values = explainer.shap_values(X_sample)
                shap_values_class0 = get_class_shap_values(shap_values, 0)
                
                col_shap1, col_shap2 = st.columns(2)
                
                with col_shap1:
                    st.markdown("**SHAP Global Beeswarm Summary (Dropout)**")
                    fig_shap, ax = plt.subplots(figsize=(8, 6))
                    shap.summary_plot(shap_values_class0, X_sample, show=False)
                    fig_shap.patch.set_facecolor('none')
                    ax.set_facecolor('none')
                    ax.tick_params(colors='#ffffff', labelsize=8)
                    ax.xaxis.label.set_color('#ffffff')
                    ax.yaxis.label.set_color('#ffffff')
                    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_color('#ffffff')
                    st.pyplot(fig_shap, clear_figure=True)
                    
                    # Download plot button
                    buf = io.BytesIO()
                    fig_shap.savefig(buf, format="png", bbox_inches='tight', facecolor='none')
                    buf.seek(0)
                    st.download_button(
                        label="📥 Download Summary Beeswarm Plot",
                        data=buf,
                        file_name="global_shap_beeswarm.png",
                        mime="image/png"
                    )
                    
                with col_shap2:
                    st.markdown("**Mean Absolute Feature Importances**")
                    mean_shap = np.abs(shap_values_class0).mean(axis=0)
                    shap_imp_df = pd.DataFrame({'Feature': X_sample.columns, 'SHAP Value': mean_shap}).sort_values('SHAP Value', ascending=False).head(12)
                    
                    fig_bar_shap = px.bar(shap_imp_df, x='SHAP Value', y='Feature', orientation='h', color='SHAP Value', color_continuous_scale='Purples')
                    fig_bar_shap.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff', yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_bar_shap, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Failed to load SHAP global explainers: {e}")
                
        st.markdown('</div>', unsafe_allow_html=True)


    # 4. EARLY WARNING SYSTEM PAGE
    elif st.session_state.navigation == "⚠️ Early Warning System":
        st.markdown('<h1 class="main-title">⚠️ Early Warning & Triage Directory</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Filter and analyze student risk directories in real-time. Review AI intervention dossiers and process batch files.</p>', unsafe_allow_html=True)
        
        # Construct full test results
        test_df = X_test.copy()
        test_df['Dropout_Probability'] = probs
        test_df['Risk_Level'] = test_df['Dropout_Probability'].apply(
            lambda p: 'Critical' if p >= 0.90 else ('High' if p >= 0.70 else ('Medium' if p >= 0.40 else 'Low'))
        )
        test_df['Student_ID'] = test_df.index + 10000
        
        c_purple = (test_df['Risk_Level'] == 'Critical').sum()
        c_red = (test_df['Risk_Level'] == 'High').sum()
        c_orange = (test_df['Risk_Level'] == 'Medium').sum()
        c_green = (test_df['Risk_Level'] == 'Low').sum()
        
        # Risk Triage Row
        col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)
        with col_c1:
            st.markdown(f'<div class="glass-card" style="padding:16px; border-left: 4px solid #667eea;"><p style="margin:0; font-size:0.8rem; color:rgba(255,255,255,0.6);">Total Directory</p><h3 style="margin:5px 0 0 0; font-size:1.8rem;">{len(test_df)}</h3></div>', unsafe_allow_html=True)
        with col_c2:
            st.markdown(f'<div class="glass-card" style="padding:16px; border-left: 4px solid #764ba2;"><p style="margin:0; font-size:0.8rem; color:rgba(255,255,255,0.6);">🟣 Critical (>=90%)</p><h3 style="margin:5px 0 0 0; font-size:1.8rem; color:#f093fb;">{c_purple}</h3></div>', unsafe_allow_html=True)
        with col_c3:
            st.markdown(f'<div class="glass-card" style="padding:16px; border-left: 4px solid #ef4444;"><p style="margin:0; font-size:0.8rem; color:rgba(255,255,255,0.6);">🔴 High (70-89%)</p><h3 style="margin:5px 0 0 0; font-size:1.8rem; color:#f87171;">{c_red}</h3></div>', unsafe_allow_html=True)
        with col_c4:
            st.markdown(f'<div class="glass-card" style="padding:16px; border-left: 4px solid #f59e0b;"><p style="margin:0; font-size:0.8rem; color:rgba(255,255,255,0.6);">🟡 Medium (40-69%)</p><h3 style="margin:5px 0 0 0; font-size:1.8rem; color:#fbbf24;">{c_orange}</h3></div>', unsafe_allow_html=True)
        with col_c5:
            st.markdown(f'<div class="glass-card" style="padding:16px; border-left: 4px solid #10b981;"><p style="margin:0; font-size:0.8rem; color:rgba(255,255,255,0.6);">🟢 Low (<40%)</p><h3 style="margin:5px 0 0 0; font-size:1.8rem; color:#34d399;">{c_green}</h3></div>', unsafe_allow_html=True)
            
        ews_tab1, ews_tab2 = st.tabs(["📋 Student Risk Directory", "📤 CSV Batch Prediction"])
        
        with ews_tab1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            # Interactive Filters
            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            with col_f1:
                f_risk = st.multiselect("Filter Risk Level:", ['Critical', 'High', 'Medium', 'Low'], default=['Critical', 'High'])
            with col_f2:
                courses = sorted(list(test_df['Course'].unique()))
                f_course = st.multiselect("Filter Course:", courses)
            with col_f3:
                f_gender = st.selectbox("Gender:", ["All", "Male", "Female"])
            with col_f4:
                f_schol = st.selectbox("Scholarship Status:", ["All", "Has Scholarship", "No Scholarship"])
                
            # Set default search query if selected from dashboard
            default_search = ""
            if st.session_state.selected_student_id is not None:
                default_search = str(st.session_state.selected_student_id)
                # clear it after read
                st.session_state.selected_student_id = None
                
            search_query = st.text_input("🔍 Search Student ID:", value=default_search)
            
            # Filtering rules
            filtered_df = test_df.copy()
            if f_risk:
                filtered_df = filtered_df[filtered_df['Risk_Level'].isin(f_risk)]
            if f_course:
                filtered_df = filtered_df[filtered_df['Course'].isin(f_course)]
            if f_gender != "All":
                g_val = 1 if f_gender == "Male" else 0
                filtered_df = filtered_df[filtered_df['Gender'] == g_val]
            if f_schol != "All":
                s_val = 1 if f_schol == "Has Scholarship" else 0
                filtered_df = filtered_df[filtered_df['Scholarship holder'] == s_val]
            if search_query:
                try:
                    filtered_df = filtered_df[filtered_df['Student_ID'].astype(str).str.contains(search_query)]
                except Exception:
                    pass
                    
            st.write(f"Displaying {len(filtered_df)} matched student records.")
            
            display_cols = ['Student_ID', 'Age at enrollment', 'Course', 'Curricular units 1st sem (grade)', 'Dropout_Probability', 'Risk_Level']
            presentation_df = filtered_df[display_cols].copy()
            presentation_df['Dropout_Probability'] = presentation_df['Dropout_Probability'].apply(lambda x: f"{x:.1%}")
            
            # Display colored dataframe
            def style_risk(val):
                if val == 'Critical':
                    return 'background-color: rgba(118, 75, 162, 0.2); color: #f093fb; font-weight: bold;'
                elif val == 'High':
                    return 'background-color: rgba(239, 68, 68, 0.2); color: #f87171; font-weight: bold;'
                elif val == 'Medium':
                    return 'background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; font-weight: bold;'
                elif val == 'Low':
                    return 'background-color: rgba(16, 185, 129, 0.2); color: #34d399; font-weight: bold;'
                return ''
                
            st.dataframe(presentation_df.style.map(style_risk, subset=['Risk_Level']), use_container_width=True)
            
            # Detail display section
            if len(filtered_df) > 0:
                st.markdown("### 🔍 AI Intervention Dossier")
                selected_st_id = st.selectbox("Select Student to View AI Dossier:", filtered_df['Student_ID'].tolist())
                st_row = filtered_df[filtered_df['Student_ID'] == selected_st_id].iloc[0]
                
                # Dynamic approximate SHAP impact dict for explanations
                shap_val_dict = {}
                try:
                    rf_model = fitted_models['Random Forest']
                    # Approximate local SHAP using feature values relative to median scaled by RF importance
                    for idx, feat in enumerate(X_test.columns):
                        val = st_row[feat]
                        median_val = X_train[feat].median()
                        direction = 1 if val < median_val else -1
                        if 'Tuition' in feat or 'Debtor' in feat: # reverse direction
                            direction = -direction
                        shap_val_dict[feat] = rf_model.feature_importances_[idx] * direction
                except Exception:
                    pass
                
                why_txt = generate_natural_language_explanation(st_row, shap_val_dict)
                recs = get_actionable_recommendations(st_row, st_row['Dropout_Probability'])
                
                col_rec1, col_rec2 = st.columns(2)
                with col_rec1:
                    st.markdown("#### 🧠 Explainable AI Insight")
                    st.write(why_txt)
                with col_rec2:
                    st.markdown("#### 💡 Targeted Intervention Recommendations")
                    for r in recs:
                        st.write(r)
                        
                    # PDF Download Button
                    try:
                        pdf_data = generate_student_pdf_report(
                            selected_st_id, 
                            st_row, 
                            st_row['Dropout_Probability'], 
                            st_row['Risk_Level'], 
                            recs, 
                            why_txt
                        )
                        st.download_button(
                            label=f"📥 Download Intervention PDF Report for #{selected_st_id}",
                            data=bytes(pdf_data),
                            file_name=f"student_intervention_report_{selected_st_id}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Error preparing PDF: {e}")
            else:
                st.info("No student matches found under current filter selection.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with ews_tab2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h4 style="margin:0 0 15px 0;">📤 CSV Batch Prediction</h4>', unsafe_allow_html=True)
            st.write("Upload a batch CSV file containing multiple students with standard features. The system will predict outcomes, categorize risk, and generate a downloadable report.")
            
            batch_file = st.file_uploader("Upload Batch CSV for inference:", type=['csv'], key="batch_prediction_uploader")
            if batch_file is not None:
                try:
                    batch_df = pd.read_csv(batch_file)
                    batch_df_eng = engineer_features(batch_df)
                    X_batch = preprocess_inference_data(batch_df_eng, X_train.columns)
                    
                    if st.session_state.selected_model_name == 'Logistic Regression':
                        batch_probs = model.predict_proba(scaler.transform(X_batch))[:, 0]
                    else:
                        batch_probs = model.predict_proba(X_batch)[:, 0]
                        
                    batch_df['Dropout_Probability'] = batch_probs
                    batch_df['Risk_Level'] = batch_df['Dropout_Probability'].apply(
                        lambda p: 'Critical' if p >= 0.90 else ('High' if p >= 0.70 else ('Medium' if p >= 0.40 else 'Low'))
                    )
                    
                    st.success("✅ Batch prediction complete!")
                    
                    col_b1, col_b2, col_b3 = st.columns(3)
                    with col_b1:
                        st.metric("Total Predicted Rows", len(batch_df))
                    with col_b2:
                        st.metric("Critical Risks", (batch_df['Risk_Level']=='Critical').sum())
                    with col_b3:
                        st.metric("High Risks", (batch_df['Risk_Level']=='High').sum())
                        
                    # Display preview
                    st.dataframe(batch_df[['Dropout_Probability', 'Risk_Level'] + list(X_batch.columns[:4])], use_container_width=True)
                    
                    csv_data = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Prediction Results CSV",
                        data=csv_data,
                        file_name="batch_predictions_results.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error performing batch prediction: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)


    # 5. PREDICTOR PAGE
    elif st.session_state.navigation == "🔮 Predictor":
        st.markdown('<h1 class="main-title">🔮 Real-time Risk Predictor</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Enter individual parameters to calculate risk probabilities, generate SHAP & LIME charts, and compile AI recommendations.</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            st.write("**👤 Demographic Indicators**")
            gender = st.selectbox("Gender:", ["Female", "Male"], index=0)
            age = st.slider("Age at enrollment:", 17, 65, 20)
            marital = st.selectbox("Marital status:", ["Single", "Married", "Divorced", "Other"], index=0)
            displaced = st.checkbox("Displaced (Commuting / Relocated)", value=False)
            needs = st.checkbox("Special Educational Needs", value=False)
        with col_in2:
            st.write("**💰 Financial Status**")
            debtor = st.checkbox("Institutional Debtor (Outstanding fees)", value=False)
            tuition = st.checkbox("Tuition Fees Paid Up to Date", value=True)
            scholarship = st.checkbox("Scholarship Holder", value=False)
            unemployment = st.slider("Unemployment rate (%):", 0.0, 20.0, 11.0)
        with col_in3:
            st.write("**📚 Academic Performance**")
            course_code = st.number_input("Course Code:", 9000, 9999, 9238)
            approved1 = st.slider("Approved units (1st sem):", 0, 15, 5)
            enrolled1 = st.slider("Enrolled units (1st sem):", 1, 15, 6)
            grade1 = st.slider("Average grade (1st sem):", 0.0, 20.0, 12.5)
            approved2 = st.slider("Approved units (2nd sem):", 0, 15, 5)
            enrolled2 = st.slider("Enrolled units (2nd sem):", 1, 15, 6)
            grade2 = st.slider("Average grade (2nd sem):", 0.0, 20.0, 12.0)

        # Map Form variables
        gender_val = 1 if gender == "Male" else 0
        marital_mapping = {"Single": 1, "Married": 2, "Divorced": 3, "Other": 4}
        marital_val = marital_mapping[marital]
        
        input_dict = {
            'Marital status': marital_val,
            'Application mode': 1,
            'Application order': 1,
            'Course': course_code,
            'Daytime/evening attendance': 1,
            'Previous qualification': 1,
            'Previous qualification (grade)': 120.0,
            'Nacionality': 1,
            'Mother\'s qualification': 10,
            'Father\'s qualification': 10,
            'Mother\'s occupation': 5,
            'Father\'s occupation': 5,
            'Admission grade': 120.0,
            'Displaced': 1 if displaced else 0,
            'Educational special needs': 1 if needs else 0,
            'Debtor': 1 if debtor else 0,
            'Tuition fees up to date': 1 if tuition else 0,
            'Gender': gender_val,
            'Scholarship holder': 1 if scholarship else 0,
            'Age at enrollment': age,
            'International': 0,
            'Curricular units 1st sem (credited)': 0,
            'Curricular units 1st sem (enrolled)': enrolled1,
            'Curricular units 1st sem (evaluations)': approved1 + 2,
            'Curricular units 1st sem (approved)': approved1,
            'Curricular units 1st sem (grade)': grade1,
            'Curricular units 1st sem (without evaluations)': 0,
            'Curricular units 2nd sem (credited)': 0,
            'Curricular units 2nd sem (enrolled)': enrolled2,
            'Curricular units 2nd sem (evaluations)': approved2 + 2,
            'Curricular units 2nd sem (approved)': approved2,
            'Curricular units 2nd sem (grade)': grade2,
            'Curricular units 2nd sem (without evaluations)': 0,
            'Unemployment rate': unemployment,
            'Inflation rate': 1.0,
            'GDP': 0.0
        }
        
        # Calculate features
        input_dict['Credit_Completion_Rate_1st_Sem'] = approved1 / enrolled1
        input_dict['Credit_Completion_Rate_2nd_Sem'] = approved2 / enrolled2
        input_dict['Grade_Drop'] = grade1 - grade2
        input_dict['Engagement_Score'] = ((approved1 / enrolled1) + (approved2 / enrolled2)) / 2.0
        input_dict['Academic_Stress_Index'] = float((enrolled1 + enrolled2) - (approved1 + approved2))
        input_dict['Financial_Risk'] = float(debtor + (1.0 - tuition) + (1.0 - scholarship))
        
        if st.button("🔮 Predict Student Outcome", type="primary"):
            st.markdown("---")
            input_df = pd.DataFrame([input_dict])
            input_df = input_df[X_train.columns]
            
            if st.session_state.selected_model_name == 'Logistic Regression':
                input_scaled = scaler.transform(input_df)
                probs = model.predict_proba(input_scaled)[0]
            else:
                probs = model.predict_proba(input_df)[0]
                
            pred_class = np.argmax(probs)
            pred_prob_dropout = probs[0]
            risk_level = 'Critical' if pred_prob_dropout >= 0.90 else ('High' if pred_prob_dropout >= 0.70 else ('Medium' if pred_prob_dropout >= 0.40 else 'Low'))
            
            col_gauge, col_text = st.columns([1, 1.2])
            with col_gauge:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = pred_prob_dropout * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Dropout Risk Probability (%)", 'font': {'color': '#ffffff'}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickcolor': "#ffffff"},
                        'bar': {'color': "#764ba2"},
                        'steps': [
                            {'range': [0, 40], 'color': "rgba(16, 185, 129, 0.15)"},
                            {'range': [40, 70], 'color': "rgba(245, 158, 11, 0.15)"},
                            {'range': [70, 90], 'color': "rgba(239, 68, 68, 0.15)"},
                            {'range': [90, 100], 'color': "rgba(118, 75, 162, 0.25)"}
                        ],
                        'threshold': {
                            'line': {'color': "#f5576c", 'width': 4},
                            'thickness': 0.8,
                            'value': 70
                        }
                    }
                ))
                fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#ffffff"}, height=260)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            with col_text:
                st.markdown(f"### Classification Rating: **{risk_level.upper()} RISK**")
                st.write(f"The model predicted the student outcome is most likely **{REVERSE_TARGET_MAPPING[pred_class].upper()}**.")
                st.write(f"**Confidence Probabilities:**")
                st.markdown(f"- Dropout Probability: `{probs[0]:.2%}`")
                st.markdown(f"- Graduate Success Probability: `{probs[1]:.2%}`")
                st.markdown(f"- Enrolled Probability: `{probs[2]:.2%}`")
                
            # Explainability Section
            st.markdown("### 🔍 Why did the model make this prediction?")
            
            exp_tab1, exp_tab2 = st.tabs(["📊 LIME Decision Explanations", "🧬 SHAP Local Attribution"])
            
            with exp_tab1:
                with st.spinner("🔄 Compiling LIME decision rules..."):
                    try:
                        lime_explainer = get_lime_explainer(X_train)
                        exp_lime = explain_with_lime(lime_explainer, model, input_df, scaler, st.session_state.selected_model_name)
                        exp_list = exp_lime.as_list(label=0)
                        
                        df_lime = pd.DataFrame(exp_list, columns=['Rule', 'Weight'])
                        df_lime['Color'] = df_lime['Weight'].apply(lambda w: 'Dropout Risk Driver' if w > 0 else 'Retention Factor')
                        
                        fig_lime = px.bar(
                            df_lime,
                            x='Weight',
                            y='Rule',
                            orientation='h',
                            color='Color',
                            color_discrete_map={
                                'Dropout Risk Driver': '#f5576c',
                                'Retention Factor': '#10b981'
                            },
                            title="LIME Decision Rules (Dropout)"
                        )
                        fig_lime.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#ffffff',
                            yaxis={'categoryorder': 'total ascending'},
                            margin=dict(t=40, b=10, l=10, r=10),
                            height=320
                        )
                        st.plotly_chart(fig_lime, use_container_width=True)
                        
                        # Native LIME HTML inside expander
                        with st.expander("View Native LIME HTML Interface"):
                            st.components.v1.html(exp_lime.as_html(), height=400, scrolling=True)
                    except Exception as e:
                        st.error(f"Failed LIME explanation: {e}")
                        
            with exp_tab2:
                with st.spinner("🔄 Running local SHAP Explainer..."):
                    try:
                        shap_explainer = get_shap_explainer(model, X_train, st.session_state.selected_model_name, scaler)
                        if shap_explainer is not None:
                            if st.session_state.selected_model_name == 'Logistic Regression':
                                s_val = shap_explainer.shap_values(scaler.transform(input_df))
                            else:
                                s_val = shap_explainer.shap_values(input_df)
                                
                            s_val_c0 = get_class_shap_values(s_val, 0).flatten()
                            
                            df_shap = pd.DataFrame({
                                'Feature': X_train.columns.tolist(),
                                'Contribution': s_val_c0.tolist()
                            })
                            df_shap['abs_contrib'] = df_shap['Contribution'].abs()
                            df_shap = df_shap.sort_values('abs_contrib', ascending=False).head(10)
                            df_shap['Color'] = df_shap['Contribution'].apply(lambda c: 'Dropout Risk Driver' if c > 0 else 'Retention Factor')
                            
                            fig_shap_local = px.bar(
                                df_shap,
                                x='Contribution',
                                y='Feature',
                                orientation='h',
                                color='Color',
                                color_discrete_map={
                                    'Dropout Risk Driver': '#f5576c',
                                    'Retention Factor': '#10b981'
                                },
                                title="Local SHAP Feature Contributions"
                            )
                            fig_shap_local.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font_color='#ffffff',
                                yaxis={'categoryorder': 'total ascending'},
                                margin=dict(t=40, b=10, l=10, r=10),
                                height=320
                            )
                            st.plotly_chart(fig_shap_local, use_container_width=True)
                        else:
                            st.warning("SHAP calculation not supported for current model config.")
                    except Exception as e:
                        st.error(f"Failed SHAP local calculations: {e}")
                        
            # Text Insight & Recommendations
            st.markdown("### 💡 Natural Language & Recommendations")
            # Build simple shap dict for text mapping
            text_shap_dict = {}
            if 'df_shap' in locals():
                text_shap_dict = dict(zip(df_shap['Feature'], df_shap['Contribution']))
            else:
                for idx, feat in enumerate(X_train.columns):
                    val = input_dict[feat]
                    mean_val = X_train[feat].median()
                    direction = 1 if val < mean_val else -1
                    if 'Tuition' in feat or 'Debtor' in feat:
                        direction = -direction
                    text_shap_dict[feat] = direction
                    
            why_txt = generate_natural_language_explanation(input_dict, text_shap_dict)
            recs = get_actionable_recommendations(input_dict, pred_prob_dropout)
            
            c_res1, c_res2 = st.columns(2)
            with c_res1:
                st.markdown("#### **AI Risk Reason Matrix**")
                st.write(why_txt)
            with c_res2:
                st.markdown("#### **Intervention Steps**")
                for r in recs:
                    st.write(r)
                    
        st.markdown('</div>', unsafe_allow_html=True)


    # 6. REPORTS PAGE
    elif st.session_state.navigation == "📈 Reports":
        st.markdown('<h1 class="main-title">📈 Academic Reports & Dossier Exports</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Print intervention dossier summaries, download risk sheets, and access research outline copies.</p>', unsafe_allow_html=True)
        
        rep_tab1, rep_tab2 = st.tabs(["📝 Printable Student Briefs", "🔬 Patents & Research Papers"])
        
        with rep_tab1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.write("Construct a customized PDF report containing student attributes, predicted dropout ratios, local XAI rules, and priority advising recommendations.")
            
            # Select student from test set
            st_indices = list(range(len(X_test)))
            st_idx = st.selectbox("Select Student Profile Index:", st_indices, format_func=lambda x: f"Student #{x+1000} (Test Sample)")
            
            st_row = X_test.iloc[st_idx]
            
            if st.session_state.selected_model_name == 'Logistic Regression':
                prob = model.predict_proba(scaler.transform(st_row.values.reshape(1,-1)))[0, 0]
            else:
                prob = model.predict_proba(st_row.values.reshape(1,-1))[0, 0]
                
            risk_lvl = 'Critical' if prob >= 0.90 else ('High' if prob >= 0.70 else ('Medium' if prob >= 0.40 else 'Low'))
            
            # Map approximate SHAP
            rep_shap_dict = {}
            for idx, feat in enumerate(X_test.columns):
                val = st_row[feat]
                mean_val = X_train[feat].median()
                direction = 1 if val < mean_val else -1
                if 'Tuition' in feat or 'Debtor' in feat:
                    direction = -direction
                rep_shap_dict[feat] = direction
                
            why_txt = generate_natural_language_explanation(st_row, rep_shap_dict)
            recs = get_actionable_recommendations(st_row, prob)
            
            if st.button("Generate Printable PDF Brief"):
                try:
                    pdf_bytes = generate_student_pdf_report(
                        st_idx + 1000,
                        st_row,
                        prob,
                        risk_lvl,
                        recs,
                        why_txt
                    )
                    st.download_button(
                        label=f"📥 Download PDF Report for Student #{st_idx+1000}",
                        data=bytes(pdf_bytes),
                        file_name=f"student_intervention_dossier_{st_idx+1000}.pdf",
                        mime="application/pdf"
                    )
                    st.success("PDF Dossier compiled!")
                except Exception as e:
                    st.error(f"Error compiling PDF Brief: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with rep_tab2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🔬 Draft Patent Abstract")
            st.code("""
TITLE: SYSTEM AND METHOD FOR REAL-TIME STUDENT DROPOUT PREDICTION AND TRIAGED EARLY WARNING SYSTEM USING EXPLAINABLE ARTIFICIAL INTELLIGENCE
            
ABSTRACT:
A system and computer-implemented method are provided to predict student dropout risk and automate targeted interventions in educational environments. The system ingests student demographics, socioeconomic factors, and academic performance metrics. A pipeline engineers composite features including a first and second semester Credit Completion Rate, a Grade Drop index, and a multidimensional Academic Stress Index. A Machine Learning comparison model handles class imbalances using Synthetic Minority Over-sampling Technique (SMOTE) to classify students into stratified risk levels (Low, Medium, High, and Critical). An Explainable AI (XAI) engine utilizes SHAP and LIME values to generate localized explanations in real-time, which are fed into a natural language compiler to construct plain-English risk driver descriptions. These descriptions automatically map features to a triaged, personalized student intervention recommendation dossier, facilitating proactive educator mediation.
            """, language="markdown")
            
            st.markdown("### 📝 Proposed IEEE Paper Outline")
            st.code("""
I. INTRODUCTION
   A. The Dropout Challenge in Higher Education
   B. Limitations of Post-Hoc Institutional Retention Reports
   C. Contributions: Proactive Early Warning System (EWS) using XAI

II. METHODOLOGY & SYSTEM ARCHITECTURE
   A. Dataset Description (UCI Student Dropout Dataset)
   B. Feature Engineering (Academic Stress Index, Grade Drop Dynamics)
   C. Synthetic Minority Over-sampling Technique (SMOTE) Implementation
   D. Classifier Architectures (Random Forest, XGBoost, LightGBM, Logistic Regression)

III. EXPLAINABLE ARTIFICIAL INTELLIGENCE (XAI)
   A. Local vs. Global Model Interpretability
   B. SHAP (Shapley Additive exPlanations) Implementation
   C. LIME (Local Interpretable Model-agnostic Explanations) Implementation
   D. Natural Language Risk Compilation Algorithm

IV. EXPERIMENTAL RESULTS
   A. Performance Evaluation and Algorithm Comparison
   B. Class Imbalance Mitigation Assessment
   C. Interpretation of Global and Local Feature Contributions

V. DISCUSSION & IMPLICATIONS FOR HIGHER EDUCATION POLICY
   A. Deploying EWS in Institutional Contexts
   B. Mapping AI Explanations to Targeted Financial & Academic Support
   C. Limitations and Future Work

VI. CONCLUSION & REFERENCES
            """, language="markdown")
            st.markdown('</div>', unsafe_allow_html=True)


def explain_with_lime(explainer, model, instance_df, scaler, selected_model_name):
    """Local LIME explanation mapping wrapper"""
    if selected_model_name == 'Logistic Regression':
        predict_fn = lambda x: model.predict_proba(scaler.transform(x))
    else:
        predict_fn = model.predict_proba
        
    exp = explainer.explain_instance(
        data_row=instance_df.values[0],
        predict_fn=predict_fn,
        num_features=10,
        labels=(0,)
    )
    return exp


if __name__ == "__main__":
    main()
