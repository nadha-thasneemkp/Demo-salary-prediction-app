import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
import os
st.write("✅ APP VERSION: DASHBOARD v2 (", __file__, ")")
# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="AI Salary Prediction Dashboard", layout="wide")

# ============================================================
# Safe base directory (important for Streamlit Cloud)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Load artifacts safely
# ============================================================
model = joblib.load(os.path.join(BASE_DIR, "salary_model.pkl"))
skill_cols = joblib.load(os.path.join(BASE_DIR, "skill_cols.pkl"))
rmse = float(joblib.load(os.path.join(BASE_DIR, "rmse.pkl")))

df_model = pd.read_csv(os.path.join(BASE_DIR, "df_model.csv"))
results_df = pd.read_csv(os.path.join(BASE_DIR, "results_df.csv"))

# ============================================================
# Safe reset mechanism
# ============================================================
RESET_KEYS = [
    "age_text", "exp_text", "usd_to_inr",
    "gender_sel", "edu_sel", "job_sel",
    "skills_selected",
    "pred", "low", "high", "pred_inr", "low_inr", "high_inr",
    "recognized", "ignored",
    "did_predict"
]

if "do_reset" not in st.session_state:
    st.session_state.do_reset = False

if st.session_state.do_reset:
    for k in RESET_KEYS:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state.do_reset = False
    st.rerun()

if "did_predict" not in st.session_state:
    st.session_state.did_predict = False


# ============================================================
# Styling
# ============================================================
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg,#eaf3ff 0%,#d8ecff 45%,#eef7ff 100%); }

section[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#0b2a66,#1d4ed8);
    color:white;
    border-right:2px solid rgba(255,255,255,0.2);
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown{
    color:white !important;
    font-weight:800 !important;
}

section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea{
    background:white !important;
    color:black !important;
    font-size:18px !important;
    font-weight:800 !important;
    border-radius:10px !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div{
    background:white !important;
    border-radius:10px !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] span{
    color:black !important;
    font-size:17px !important;
    font-weight:800 !important;
}

section[data-testid="stSidebar"] button{
    background:linear-gradient(90deg,#3b82f6,#60a5fa) !important;
    color:white !important;
    font-weight:900 !important;
    border-radius:12px !important;
    width:100% !important;
}

.panel{
    background:white;
    border-radius:16px;
    padding:16px;
    box-shadow:0 10px 26px rgba(0,0,0,0.10);
}

.kpi{
    background:white;
    border-radius:14px;
    padding:16px;
    box-shadow:0 8px 20px rgba(0,0,0,0.08);
}

.kpi .label{ font-size:13px; color:#475569; font-weight:800; }
.kpi .value{ font-size:30px; font-weight:900; color:#0f172a; }

.skillbox-ok{
    background: linear-gradient(180deg,#dcfce7,#bbf7d0);
    border:2px solid #22c55e;
    border-radius:16px;
    padding:26px;
    font-size:19px;
    font-weight:800;
}

.skillbox-warn{
    background: linear-gradient(180deg,#fffbeb,#fde68a);
    border:2px solid #f59e0b;
    border-radius:16px;
    padding:26px;
    font-size:19px;
    font-weight:800;
}

.small{ color:#334155; font-size:13px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Helper functions
# ============================================================
SKILL_CANONICAL = {
    "sql": "SQL",
    "api": "APIs",
    "apis": "APIs",
    "power bi": "Power BI",
    "powerbi": "Power BI",
    "ml": "Machine Learning",
    "dl": "Deep Learning",
}

def normalize_job_title(t):
    t = str(t).strip()
    t = re.sub(r"\s+", " ", t)
    return t.title()

def normalize_skill_name(s):
    s2 = str(s).strip()
    low = s2.lower()
    return SKILL_CANONICAL.get(low, s2)


# ============================================================
# Graphs
# ============================================================
def plot_salary_vs_experience(df):
    tmp = df.copy()
    tmp["Years of Experience"] = tmp["Years of Experience"].round().astype(int)
    grp = tmp.groupby("Years of Experience")["Salary"].median().reset_index()

    fig = plt.figure(figsize=(7,4))
    plt.plot(grp["Years of Experience"], grp["Salary"])
    plt.title("Median Salary vs Experience")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.tight_layout()
    return fig


def plot_avg_salary_by_role(df):
    grp = df.groupby("Job Title")["Salary"].mean().sort_values(ascending=False).head(10)

    fig = plt.figure(figsize=(7,4))
    plt.bar(grp.index, grp.values)
    plt.title("Average Salary by Role")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig


# ============================================================
# Header
# ============================================================
st.title("AI Salary Prediction Dashboard")
st.write('<div class="small">Predict salary and visualize insights.</div>', unsafe_allow_html=True)


# ============================================================
# Sidebar Inputs
# ============================================================
with st.sidebar:
    st.subheader("User Inputs")

    gender_options = ["Select Gender","Male","Female","Other"]
    edu_options = ["Select Education","High School","Bachelor","Master","PhD"]

    if df_model is not None and "Job Title" in df_model.columns:
        job_titles = sorted(df_model["Job Title"].dropna().unique())
        job_options = ["Select Job Title"] + job_titles
    else:
        job_options = ["Select Job Title"]

    with st.form("predict_form"):

        age_text = st.text_input("Age",key="age_text")
        gender_sel = st.selectbox("Gender",gender_options,key="gender_sel")
        edu_sel = st.selectbox("Education",edu_options,key="edu_sel")
        job_sel = st.selectbox("Job Title",job_options,key="job_sel")
        exp_text = st.text_input("Years of Experience",key="exp_text")

        st.multiselect("Select Skills",options=sorted(skill_cols),key="skills_selected")
        usd_to_inr = st.number_input("USD → INR Rate",value=83.0,key="usd_to_inr")

        predict_btn = st.form_submit_button("Predict")
        reset_btn = st.form_submit_button("Reset")

    if reset_btn:
        st.session_state.do_reset=True
        st.rerun()


# ============================================================
# Prediction logic
# ============================================================
if predict_btn:

    age_val=int(age_text)
    exp_val=float(exp_text)

    skills_selected=st.session_state.get("skills_selected",[])
    skills_list=sorted(set([normalize_skill_name(s) for s in skills_selected]))

    row={
        "Age":float(age_val),
        "Years of Experience":float(exp_val),
        "Gender":gender_sel,
        "Education Level":edu_sel,
        "Job Title":normalize_job_title(job_sel)
    }

    for sc in skill_cols:
        row[sc]=1 if sc in skills_list else 0

    X_user=pd.DataFrame([row])
    pred=float(model.predict(X_user)[0])

    low=max(0,pred-rmse)
    high=pred+rmse

    rate=float(usd_to_inr)
    pred_inr=pred*rate
    low_inr=low*rate
    high_inr=high*rate

    st.success(f"Predicted Salary (USD): ${pred:,.0f}")
    st.info(f"Range: ${low:,.0f} – ${high:,.0f}")

    st.success(f"Predicted Salary (INR): ₹{pred_inr:,.0f}")
    st.info(f"Range: ₹{low_inr:,.0f} – ₹{high_inr:,.0f}")


# ============================================================
# Graphs
# ============================================================
if st.session_state.did_predict and df_model is not None:
    st.pyplot(plot_salary_vs_experience(df_model))

    st.pyplot(plot_avg_salary_by_role(df_model))

