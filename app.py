import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Data Cleaning", layout="wide")

# -------------------------------
# SIMPLE STYLING
# -------------------------------
st.markdown("""
    <style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.title("🧠 AI Data Cleaning Environment")

# -------------------------------
# FILE UPLOAD
# -------------------------------
st.subheader("📂 Upload Your Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # -------------------------------
    # ISSUE DETECTION
    # -------------------------------
    def detect_issues(df):
        missing = df.isnull().sum().sum()
        negative = (df.select_dtypes(include=[np.number]) < 0).sum().sum()
        outliers = ((df.select_dtypes(include=[np.number]) > 100)).sum().sum()
        return {
            "missing": int(missing),
            "negative": int(negative),
            "outliers": int(outliers)
        }

    issues = detect_issues(df)

    # -------------------------------
    # SCORE CALCULATION
    # -------------------------------
    score = 100 - (issues["missing"] * 2 + issues["negative"] * 3 + issues["outliers"] * 2)
    score = max(score, 0)

    # -------------------------------
    # METRICS
    # -------------------------------
    st.subheader("📊 Data Quality Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Missing Values", issues["missing"])
    col2.metric("Errors", issues["negative"] + issues["outliers"])
    col3.metric("Score", score)

    # -------------------------------
    # CHART
    # -------------------------------
    chart_data = pd.DataFrame({
        "Category": ["Missing", "Errors"],
        "Count": [issues["missing"], issues["negative"] + issues["outliers"]]
    })

    st.bar_chart(chart_data.set_index("Category"))

    # -------------------------------
    # CLEANING ACTIONS
    # -------------------------------
    st.subheader("⚙️ Cleaning Actions")

    action = st.selectbox("Choose Action", [
        "None",
        "Fill Missing (Mean)",
        "Remove Negative Values",
        "Remove Outliers"
    ])

    if st.button("Apply Action"):
        if action == "Fill Missing (Mean)":
            df.fillna(df.mean(numeric_only=True), inplace=True)

        elif action == "Remove Negative Values":
            df[df.select_dtypes(include=[np.number]).columns] = df.select_dtypes(include=[np.number]).clip(lower=0)

        elif action == "Remove Outliers":
            df[df.select_dtypes(include=[np.number]).columns] = df.select_dtypes(include=[np.number]).clip(upper=100)

        st.success("✅ Action Applied Successfully!")

    # -------------------------------
    # AI AUTO CLEAN BUTTON
    # -------------------------------
    if st.button("🤖 Run AI Auto Clean"):
        df.fillna(df.mean(numeric_only=True), inplace=True)
        df[df.select_dtypes(include=[np.number]).columns] = df.select_dtypes(include=[np.number]).clip(lower=0)
        df[df.select_dtypes(include=[np.number]).columns] = df.select_dtypes(include=[np.number]).clip(upper=100)

        st.success("🤖 AI cleaned the dataset automatically!")

    # -------------------------------
    # FINAL DATA
    # -------------------------------
    st.subheader("📈 Cleaned Dataset")
    st.dataframe(df, use_container_width=True)

else:
    st.info("👆 Please upload a CSV file to start")
