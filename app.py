import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt

# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="AI Salary Prediction Dashboard", layout="wide")

# ============================================================
# Load artifacts
# ============================================================
model = joblib.load("salary_model.pkl")
skill_cols = joblib.load("skill_cols.pkl")
rmse = float(joblib.load("rmse.pkl"))

df_model = None
results_df = None
try:
    df_model = pd.read_csv("df_model.csv")
except:
    df_model = None

try:
    results_df = pd.read_csv("results_df.csv")
except:
    results_df = None

# ============================================================
# Safe reset mechanism (IMPORTANT)
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

# If user requested reset in previous run, clear BEFORE widgets are created
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
    height:100%;
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
.skillbox-ok h4, .skillbox-warn h4{
    margin: 0 0 10px 0;
    font-size:20px;
    font-weight:900;
}

.small{ color:#334155; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Helpers
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
def normalize_job_title(t: str) -> str:
    t = str(t).strip()
    t = re.sub(r"\s+", " ", t)
    return t.title()

def normalize_skill_name(s: str) -> str:
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

    fig = plt.figure(figsize=(7, 4))
    plt.plot(grp["Years of Experience"], grp["Salary"])
    plt.title("Median Salary vs Experience")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.tight_layout()
    return fig

def plot_avg_salary_by_role(df, top_n=10):
    grp = df.groupby("Job Title")["Salary"].mean().sort_values(ascending=False).head(top_n)
    fig = plt.figure(figsize=(7, 4))
    plt.bar(grp.index, grp.values)
    plt.title(f"Average Salary by Role (Top {top_n})")
    plt.ylabel("Average Salary")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig

def plot_feature_importance(model, top_n=15):
    try:
        preprocess = model.named_steps["preprocess"]
        m = model.named_steps["model"]
        feat_names = preprocess.get_feature_names_out()
        importances = getattr(m, "feature_importances_", None)

        fig = plt.figure(figsize=(7.5, 4.8))
        if importances is None:
            plt.text(0.05, 0.5, "Feature importance not available for this model.", fontsize=12)
            plt.axis("off")
            plt.tight_layout()
            return fig

        importances = np.array(importances)
        idx = np.argsort(importances)[::-1][:top_n]

        plt.barh(range(len(idx))[::-1], importances[idx][::-1])
        plt.yticks(range(len(idx))[::-1], [str(feat_names[i]) for i in idx][::-1], fontsize=9)
        plt.title(f"Top {top_n} Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        return fig
    except Exception:
        fig = plt.figure(figsize=(7, 4))
        plt.text(0.05, 0.5, "Could not compute feature importance.", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        return fig

def plot_model_comparison(results_df):
    fig = plt.figure(figsize=(7.2, 4.2))
    if results_df is None or len(results_df) == 0:
        plt.text(0.05, 0.5, "Place results_df.csv in folder to show model comparison.", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        return fig

    dfp = results_df.copy()
    if "val_R2" in dfp.columns:
        dfp = dfp.sort_values("val_R2", ascending=False)
        plt.bar(dfp["model"], dfp["val_R2"])
        plt.ylim(0, 1.0)
        plt.title("Model Comparison (Validation R²)")
        plt.ylabel("Validation R²")
        plt.xticks(rotation=30, ha="right")
    else:
        dfp = dfp.sort_values("val_RMSE", ascending=True)
        plt.bar(dfp["model"], dfp["val_RMSE"])
        plt.title("Model Comparison (Validation RMSE)")
        plt.ylabel("Validation RMSE")
        plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    return fig

# ============================================================
# Header
# ============================================================
st.title("AI Salary Prediction Dashboard")
st.write('<div class="small">Predict annual salary (USD) and show INR conversion. Graphs appear after prediction.</div>', unsafe_allow_html=True)

# ============================================================
# Sidebar Inputs (placeholders)
# ============================================================
with st.sidebar:
    st.subheader("User Inputs")

    # placeholders via selectbox with a "Select ..." item
    gender_options = ["Select Gender", "Male", "Female", "Other"]
    edu_options = ["Select Education", "High School", "Bachelor", "Master", "PhD"]

    # job titles list (optional)
    if df_model is not None and "Job Title" in df_model.columns:
        job_titles = sorted(df_model["Job Title"].dropna().unique())
        job_options = ["Select Job Title"] + job_titles
    else:
        job_options = ["Select Job Title"]  # fallback

    with st.form("predict_form", clear_on_submit=False):
        age_text = st.text_input("Age", placeholder="Enter Age", key="age_text")
        gender_sel = st.selectbox("Gender", gender_options, key="gender_sel")
        edu_sel = st.selectbox("Education", edu_options, key="edu_sel")
        job_sel = st.selectbox("Job Title", job_options, key="job_sel")
        exp_text = st.text_input("Years of Experience", placeholder="Enter Experience (e.g., 3 or 3.5)", key="exp_text")

        st.multiselect("Select Skills (multiple)", options=sorted(skill_cols), key="skills_selected")
        usd_to_inr = st.number_input("USD → INR Rate", value=83.0, key="usd_to_inr")

        c1, c2 = st.columns(2)
        with c1:
            predict_btn = st.form_submit_button("Predict")
        with c2:
            reset_btn = st.form_submit_button("Reset")

    if reset_btn:
        st.session_state.do_reset = True
        st.rerun()

# ============================================================
# Predict
# ============================================================
if predict_btn:
    # validation
    if not age_text.strip():
        st.sidebar.error("Please enter Age.")
        st.stop()

    try:
        age_val = int(float(age_text))
    except:
        st.sidebar.error("Age must be a number.")
        st.stop()

    if age_val < 18:
        st.sidebar.error("Age must be at least 18.")
        st.stop()

    if gender_sel == "Select Gender":
        st.sidebar.error("Please select Gender.")
        st.stop()

    if edu_sel == "Select Education":
        st.sidebar.error("Please select Education.")
        st.stop()

    if job_sel == "Select Job Title":
        st.sidebar.error("Please select Job Title.")
        st.stop()

    if not exp_text.strip():
        st.sidebar.error("Please enter Years of Experience.")
        st.stop()

    try:
        exp_val = float(exp_text)
    except:
        st.sidebar.error("Years of Experience must be a number.")
        st.stop()

    if exp_val < 0:
        st.sidebar.error("Years of Experience cannot be negative.")
        st.stop()

    # build row for model
    skills_selected = st.session_state.get("skills_selected", [])
    skills_list = sorted(set([normalize_skill_name(s) for s in skills_selected]))

    row = {
        "Age": float(age_val),
        "Years of Experience": float(exp_val),
        "Gender": gender_sel,
        "Education Level": edu_sel,
        "Job Title": normalize_job_title(job_sel),
    }
    for sc in skill_cols:
        row[sc] = 1 if sc in skills_list else 0

    X_user = pd.DataFrame([row])
    pred = float(model.predict(X_user)[0])

    low, high = max(0.0, pred - rmse), pred + rmse
    rate = float(usd_to_inr)
    pred_inr = pred * rate
    low_inr, high_inr = low * rate, high * rate

    recognized = [s for s in skills_list if s in skill_cols]
    ignored = [s for s in skills_list if s not in skill_cols]

    st.session_state.pred = pred
    st.session_state.low = low
    st.session_state.high = high
    st.session_state.pred_inr = pred_inr
    st.session_state.low_inr = low_inr
    st.session_state.high_inr = high_inr
    st.session_state.recognized = recognized
    st.session_state.ignored = ignored
    st.session_state.did_predict = True

# ============================================================
# Layout
# ============================================================
left, right = st.columns([1.6, 1])

with right:
    st.markdown('<div class="panel"><h3>Prediction Result</h3><div class="small">KPI cards + prominent skills.</div></div>', unsafe_allow_html=True)

    if not st.session_state.did_predict:
        st.info("Enter inputs in the sidebar and click **Predict**.")
    else:
        pred = st.session_state.pred
        low = st.session_state.low
        high = st.session_state.high
        pred_inr = st.session_state.pred_inr
        low_inr = st.session_state.low_inr
        high_inr = st.session_state.high_inr
        recognized = st.session_state.recognized
        ignored = st.session_state.ignored

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""<div class="kpi"><div class="label">Predicted Annual (USD)</div><div class="value">${pred:,.0f}</div></div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class="kpi"><div class="label">Range (USD)</div><div class="value">${low:,.0f} – ${high:,.0f}</div></div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""<div class="kpi"><div class="label">Monthly (USD)</div><div class="value">${pred/12:,.0f}</div></div>""", unsafe_allow_html=True)

        st.write("")

        r4, r5, r6 = st.columns(3)
        with r4:
            st.markdown(f"""<div class="kpi"><div class="label">Approx Annual (INR)</div><div class="value">₹{pred_inr:,.0f}</div></div>""", unsafe_allow_html=True)
        with r5:
            st.markdown(f"""<div class="kpi"><div class="label">Range (INR)</div><div class="value">₹{low_inr:,.0f} – ₹{high_inr:,.0f}</div></div>""", unsafe_allow_html=True)
        with r6:
            st.markdown(f"""<div class="kpi"><div class="label">Monthly (INR)</div><div class="value">₹{pred_inr/12:,.0f}</div></div>""", unsafe_allow_html=True)

        st.write("---")

        ok_text = ", ".join(recognized) if recognized else "None"
        st.markdown(f"""<div class="skillbox-ok"><h4>✅ Recognized Skills</h4>{ok_text}</div>""", unsafe_allow_html=True)

        st.write("")

        bad_text = ", ".join(ignored) if ignored else "None"
        st.markdown(f"""<div class="skillbox-warn"><h4>⚠️ Ignored Skills</h4>{bad_text}</div>""", unsafe_allow_html=True)

with left:
    st.markdown('<div class="panel"><h3>Analytics & Visualizations</h3><div class="small">Graphs appear only after prediction.</div></div>', unsafe_allow_html=True)

    if not st.session_state.did_predict:
        st.info("Make a prediction to unlock graphs.")
    else:
        tabs = st.tabs(["Market Trends", "Interpretability", "Model Comparison"])

        with tabs[0]:
            if df_model is None:
                st.warning("Add df_model.csv in SalaryApp folder to show trend graphs.")
            else:
                st.pyplot(plot_salary_vs_experience(df_model))
                st.pyplot(plot_avg_salary_by_role(df_model, top_n=10))

        with tabs[1]:
            st.pyplot(plot_feature_importance(model, top_n=15))

        with tabs[2]:
            st.pyplot(plot_model_comparison(results_df))