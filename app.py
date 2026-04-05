import streamlit as st
import pandas as pd
from backend import DataPrepEnv, AutoAgent

st.set_page_config(page_title="DataPrepEnv", layout="wide")

st.title("🚀 DataPrepEnv: AI Data Cleaning Environment")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your messy CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Initialize environment once
    if "env" not in st.session_state:
        st.session_state.env = DataPrepEnv(df)

    env = st.session_state.env

    # Show dataset
    st.subheader("📊 Current Dataset")
    st.dataframe(env.df)

    # Show issues
    st.subheader("⚠️ Data Issues")
    st.write(env.get_issues())

    # Show score
    st.subheader("📈 Data Quality Score")
    st.metric("Score", env.get_score())

    # ------------------------
    # Manual Cleaning
    # ------------------------
    st.subheader("⚙️ Apply Manual Cleaning")

    action = st.selectbox("Choose Action", [
        "fill_missing_mean",
        "fill_missing_mode",
        "remove_rows",
        "fix_negative",
        "cap_outliers"
    ])

    if st.button("▶️ Apply Action"):
        _, reward, old_score, new_score = env.step(action)

        st.success(f"Action Applied: {action}")
        st.info(f"Reward: {reward}")
        st.write(f"Score: {old_score} → {new_score}")

    # ------------------------
    # AI Agent
    # ------------------------
    st.subheader("🤖 Auto AI Cleaning")

    if st.button("🚀 Run AI Agent"):
        agent = AutoAgent(env)
        steps = agent.run()

        if not steps:
            st.warning("No further improvement possible.")
        else:
            for step in steps:
                st.write(
                    f"👉 {step['action']} | "
                    f"Reward: {step['reward']} | "
                    f"Score: {step['old_score']} → {step['new_score']}"
                )

    # Final dataset
    st.subheader("🧼 Cleaned Dataset")
    st.dataframe(env.df)

    # Download
    st.download_button(
        "📥 Download Cleaned Data",
        env.df.to_csv(index=False),
        "cleaned_data.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Upload a CSV file to start cleaning.")